"""Reflection node implementation.

One LLM call that does what cognitive_update and action_selection used to do in
two: update working memory, form new memories, and choose an action. Merging
them removes a full round-trip and the ~1,000-1,900 tokens per cycle the two
prompts duplicated (world knowledge, recent events, personality, format
scaffolding), and lets the static half of the prompt be served from the
provider's cache (NPC-1319).
"""

from pathlib import Path
from pprint import pformat

from json_repair import loads as json_repair_loads
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.nodes.base import LLMNode, entity_tag
from mind.cognitive_architecture.nodes.formatting import (
    format_interaction_status as _format_interaction_status,
)
from mind.cognitive_architecture.nodes.formatting import (
    format_personality,
    format_substrate_goal,
)
from mind.cognitive_architecture.observations import MindEvent, MindEventType
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import NewMemory, WorkingMemory
from mind.knowledge import KnowledgeBase, KnowledgeFile
from mind.logging_config import get_logger

from .models import ReflectionOutput

logger = get_logger()

# Splits prompt.md into the cacheable static prefix (everything up to and
# including this line) and the per-call dynamic suffix. Exactly one occurrence
# is required; the prompt-contract test pins that.
CACHE_BREAKPOINT_MARKER = (
    "<!-- CACHE BREAKPOINT: nothing above this line may vary between calls -->"
)


class ReflectionNode(LLMNode):
    """Updates cognitive state and selects an action in a single call.

    Named for the conscious rung of the cognition-escalation ladder: this node
    is one full pass of deliberate reflection over the substrate's state.
    """

    step_name = "reflection"

    def __init__(self, llm: BaseChatModel):
        raw = (Path(__file__).parent / "prompt.md").read_text()
        if raw.count(CACHE_BREAKPOINT_MARKER) != 1:
            raise ValueError(
                f"prompt.md must contain exactly one cache-breakpoint marker, "
                f"found {raw.count(CACHE_BREAKPOINT_MARKER)}"
            )
        static_text, dynamic_text = raw.split(CACHE_BREAKPOINT_MARKER)
        dynamic_prompt = PromptTemplate.from_template(dynamic_text)

        super().__init__(llm, dynamic_prompt, output_model=ReflectionOutput, max_retries=2)

        # The static prefix is formatted exactly ONCE, here: its two variables
        # (world knowledge, format instructions) are fixed at construction, so
        # the bytes are identical across every call and every NPC -- the
        # property the provider cache keys on. build_static_prefix raises if a
        # dynamic variable leaked above the breakpoint. Assigned after
        # super().__init__ because format instructions need the parser it sets
        # up.
        world_knowledge = KnowledgeBase.get(
            [
                KnowledgeFile.NEEDS,
                KnowledgeFile.INTERACTIONS,
                KnowledgeFile.MOVEMENT,
                KnowledgeFile.ACTIVITY,
            ]
        )
        self.static_prefix = (
            LLMNode.build_static_prefix(
                static_text,
                world_knowledge=world_knowledge,
                format_instructions=self.get_format_instructions(),
            )
            + CACHE_BREAKPOINT_MARKER
        )
        self.cache_control = True

    async def process(self, state: PipelineState) -> PipelineState:
        """Reflect once: update working memory, form memories, choose an action"""

        # Format memories for prompt
        memories_text = "\n".join([str(m) for m in state.retrieved_memories])
        if not memories_text:
            memories_text = "No relevant memories found."

        # Shared helpers keep these renderings identical across the pipeline
        personality_text, dims_text = format_personality(
            state.personality_traits, state.personality_dimensions
        )

        # Format available actions
        actions_text = "\n".join([f"- {str(action)}" for action in state.available_actions])

        output = await self.call_llm(
            state,
            # Salvage instead of raise: a parse failure must not cost the whole
            # cycle (the old cognitive_update path lost action, memory write,
            # and telemetry together -- NPC-1195).
            on_exhausted=lambda raw, error: self._salvage(state, raw, error),
            working_memory=str(state.working_memory),
            personality_traits=personality_text,
            personality_dimensions=dims_text,
            # Ground the "am I interacting?" belief in the fresh observation so
            # a stale working-memory belief gets corrected this cycle (NPC-688).
            interaction_status=_format_interaction_status(state.observation),
            substrate_goal=format_substrate_goal(
                state.observation.goal if state.observation else None
            ),
            retrieved_memories=memories_text,
            recent_events=pformat(state.recent_events),
            observation_text=str(state.observation),
            available_actions=actions_text,
        )

        # Update state with new working memory
        state.working_memory = output.updated_working_memory

        # Add new memories to daily buffer
        state.daily_memories.extend(output.new_memories)

        state.chosen_action = output.chosen_action

        # Create ACTION_CHOSEN event
        action_event = MindEvent(
            timestamp=state.observation.current_simulation_time,
            event_type=MindEventType.ACTION_CHOSEN,
            payload={
                "action": output.chosen_action.action,
                "parameters": output.chosen_action.parameters,
            },
        )
        state.recent_events.append(action_event)

        # Log updated working memory as a single record so the simulation's
        # Events tab shows one entry per thought instead of one per field
        tag = entity_tag(state)
        wm = output.updated_working_memory
        logger.debug(
            f"{tag} Updated working memory:\n"
            f"  Situation: {wm.situation_assessment}\n"
            f"  Active goals: {wm.active_goals}\n"
            f"  Current plan: {wm.current_plan}\n"
            f"  Emotional state: {wm.emotional_state}"
        )

        # Log new memories as a single record for the same reason
        memory_lines = "".join(
            f"\n  [importance={mem.importance}] {mem.content}" for mem in output.new_memories
        )
        logger.debug(f"{tag} Storing {len(output.new_memories)} new memories{memory_lines}")

        # Log action selection
        logger.debug(f"{tag} Evaluated {len(state.available_actions)} available actions")
        logger.debug(f"{tag} Selected: {output.chosen_action}")

        return state

    def _salvage(self, state: PipelineState, raw: str | None, error: Exception) -> ReflectionOutput:
        """Rescue whatever validates from the last raw response, part by part.

        Each field is salvaged independently, so a malformed action does not
        discard a good working-memory update and vice versa. The floors: the
        working memory falls back to the CURRENT one, never an empty one (an
        empty WorkingMemory validates trivially and would silently wipe the
        NPC's mind), and the action falls back to WAIT, which passes every
        branch of the pipeline's downstream re-validation.
        """
        data = json_repair_loads(raw) if raw else None
        if not isinstance(data, dict):
            data = {}

        salvaged_wm: WorkingMemory | None = None
        try:
            candidate = WorkingMemory.model_validate(data.get("updated_working_memory"))
            if str(candidate) != "No working memory":
                salvaged_wm = candidate
        except ValidationError:
            pass

        salvaged_memories: list[NewMemory] = []
        entries = data.get("new_memories")
        if isinstance(entries, list):
            for entry in entries:
                try:
                    salvaged_memories.append(NewMemory.model_validate(entry))
                except ValidationError:
                    continue

        salvaged_action: Action | None = None
        try:
            salvaged_action = Action.model_validate(
                data.get("chosen_action"), context={"state": state}
            )
        except ValidationError:
            pass

        logger.warning(
            f"{entity_tag(state)} Reflection failed after retries; salvaged "
            f"working_memory={'yes' if salvaged_wm is not None else 'no (kept current)'}, "
            f"{len(salvaged_memories)} memories, "
            f"action={'yes' if salvaged_action is not None else 'no (falling back to wait)'}: "
            f"{error}"
        )

        if salvaged_wm is None:
            salvaged_wm = state.working_memory
        if salvaged_action is None:
            # model_construct bypasses validation - WAIT is always safe
            salvaged_action = Action.model_construct(action=ActionType.WAIT, parameters={})

        return ReflectionOutput.model_construct(
            updated_working_memory=salvaged_wm,
            new_memories=salvaged_memories,
            chosen_action=salvaged_action,
        )
