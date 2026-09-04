"""Mind runtime state and behavior"""

from dataclasses import dataclass, field
from typing import Self

from mind.apis.langchain_llm import get_llm
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    MindEvent,
    MindEventType,
    Observation,
)
from mind.cognitive_architecture.pipeline import CognitivePipeline
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import NewMemory, WorkingMemory
from mind.interfaces.mcp.models import MindConfig
from mind.logging_config import get_logger

logger = get_logger()

# Event buffer retention policy
EVENT_RETENTION_TIME_MINUTES = 60  # Keep events newer than this many game minutes
EVENT_BUFFER_MAX_SIZE = 15  # Maximum number of events to retain


@dataclass
class Mind:
    """Runtime state for a single mind - encapsulates all mind behavior"""

    mind_id: str
    entity_id: str  # Mind's entity ID in simulation
    traits: list[str]
    pipeline: CognitivePipeline
    memory_store: VectorDBMemory
    working_memory: WorkingMemory

    # The model slug this mind was CONFIGURED with. Deterministic and testable
    # without a live call, which is why telemetry reports it rather than
    # whatever the provider actually served: OpenRouter may route a request to a
    # variant, so this is the requested model, not necessarily the served one.
    llm_model: str
    personality_dimensions: dict[str, float] = field(default_factory=dict)
    daily_memories: list[NewMemory] = field(default_factory=list)

    # Conversation history aggregation (keyed by interaction_id)
    conversation_histories: dict[str, list[ConversationMessage]] = field(default_factory=dict)
    _finished_conversation_ids: set[str] = field(default_factory=set, repr=False)

    event_buffer: list[MindEvent] = field(default_factory=list)

    # Pending incoming interaction bids (keyed by bid_id from payload)
    pending_incoming_bids: dict[str, MindEvent] = field(default_factory=dict)

    # Elapsed game minutes as of the last decide_action. Consolidation runs
    # outside the decision cycle (the simulation calls it on wake), so it has no
    # observation of its own and needs the mind to have remembered when "now"
    # last was - otherwise consolidated memories carry no usable timestamp and
    # the recency term is fed a constant. None until the first decision: a mind
    # can be consolidated before it has ever decided, and that is genuinely
    # "unknown", not "time zero".
    last_simulation_time: int | None = None

    @classmethod
    def from_config(cls, mind_id: str, entity_id: str, config: MindConfig) -> Self:
        """Create a Mind instance from configuration

        Args:
            mind_id: The mind's own identifier (PK) - keys the memory collection
            entity_id: The simulation entity this mind drives (FK) - deliberately
                independent of mind_id; carried for per-NPC log attribution
            config: MindConfig with traits, LLM, memory, and personality settings

        Returns:
            Initialized Mind instance
        """
        # Initialize LLM from config
        llm = get_llm(config.llm_model)

        # Memory belongs to the mind, so the collection is keyed by the mind PK
        memory_store = VectorDBMemory(
            collection_name=f"mind_{mind_id}",
            embedding_model=config.embedding_model,
            storage_path=config.memory_storage_path,
        )

        # Seed initial long-term memories. No importance and no timestamp: these
        # have never been rated by anything, and they did not happen at a moment
        # in the simulation. The previous hardcoded importance=5.0 was a
        # fabricated rating that outranked every genuinely mundane lived memory.
        for memory_content in config.initial_long_term_memories:
            memory_store.add_memory(content=memory_content)

        # Initialize pipeline
        pipeline = CognitivePipeline(
            llm=llm, memory_store=memory_store, retrieval_weights=config.retrieval_weights
        )

        # Initialize working memory
        working_memory = config.initial_working_memory or WorkingMemory()

        # Create Mind instance
        return cls(
            mind_id=mind_id,
            entity_id=entity_id,
            traits=config.traits,
            personality_dimensions=config.personality_dimensions,
            pipeline=pipeline,
            memory_store=memory_store,
            working_memory=working_memory,
            llm_model=config.llm_model,
        )

    @classmethod
    def reattach(cls, mind_id: str, entity_id: str, config: MindConfig) -> Self:
        """Re-attach a Mind to its retained memory collection.

        Identical to from_config except it does NOT seed
        config.initial_long_term_memories. The collection already exists (it was
        retained when the mind was released, not deleted), and VectorDBMemory uses
        get_or_create_collection, so this transparently reopens it. Skipping the
        seed loop is what keeps re-attaching idempotent - re-seeding here would
        duplicate the original seeds on every relink.

        Args:
            mind_id: The mind's own identifier (PK) - keys the retained collection
            entity_id: The simulation entity this mind now drives (FK). May differ
                from the entity the mind drove before release; the relink rebinds it.
            config: MindConfig with traits, LLM, memory, and personality settings.
                initial_long_term_memories is intentionally ignored here.

        Returns:
            Initialized Mind instance bound to the existing collection
        """
        # Initialize LLM from config
        llm = get_llm(config.llm_model)

        # Reopen the retained collection keyed by the mind PK (no seeding)
        memory_store = VectorDBMemory(
            collection_name=f"mind_{mind_id}",
            embedding_model=config.embedding_model,
            storage_path=config.memory_storage_path,
        )

        # Initialize pipeline. Threading retrieval_weights here as well as in
        # from_config is load-bearing: a relinked mind that silently reverted to
        # the default weights would think differently from the same mind before
        # its release, with nothing in the logs to say so (the NPC-1023 shape).
        pipeline = CognitivePipeline(
            llm=llm, memory_store=memory_store, retrieval_weights=config.retrieval_weights
        )

        # Initialize working memory
        working_memory = config.initial_working_memory or WorkingMemory()

        # Create Mind instance
        return cls(
            mind_id=mind_id,
            entity_id=entity_id,
            traits=config.traits,
            personality_dimensions=config.personality_dimensions,
            pipeline=pipeline,
            memory_store=memory_store,
            working_memory=working_memory,
            llm_model=config.llm_model,
        )

    def update_conversations(self, conversations: list) -> None:
        """Aggregate conversation updates into full history.

        Observations arrive as overlapping rolling windows, so the same message
        is re-sent on many cycles and must be stored exactly once. Identity is
        the message's ``id`` and nothing else: the simulation mints one for
        every message as a class invariant, so there is no id-less message to
        accommodate and no second keying strategy to keep alive beside this one.

        The index is rebuilt per conversation and **mutated inside the loop**.
        Building it once up front was a distinct defect from the keying one: a
        single batch carrying the same message twice stored it twice, and that
        stays true of an id-only key.

        Args:
            conversations: List of ConversationObservation objects
        """
        for conv_obs in conversations:
            interaction_id = conv_obs.interaction_id

            # Initialize if new conversation
            if interaction_id not in self.conversation_histories:
                self.conversation_histories[interaction_id] = []

            stored = self.conversation_histories[interaction_id]
            seen_ids = {msg.id for msg in stored}

            for msg in conv_obs.conversation_history:
                # Normalised once, here, so the value we VALIDATE is the value we
                # STORE and key on. Checking `.strip()` for emptiness while keying on
                # the untrimmed original would make " message_a " and "message_a" two
                # identities for one message — the test and the thing tested drifting
                # apart.
                message_id = msg.id.strip()
                if not message_id:
                    # An empty id is a producer bug: the simulation mints in the
                    # constructor and logs an error if it ever serializes a blank,
                    # so a blank arriving here means that invariant was bypassed.
                    # There is nowhere to fall through to — "" cannot be an
                    # identity, since every blank would collide with every other —
                    # so the message is refused, loudly. Loud and inert beats
                    # silently collapsing distinct messages into one.
                    logger.error(
                        f"[{self.entity_id}] Conversation message in {interaction_id} arrived "
                        f"with an EMPTY id (speaker={msg.speaker_id!r}) — the producer's mint "
                        f"invariant was bypassed. Refusing the message; it cannot be identified."
                    )
                    continue

                if message_id in seen_ids:
                    continue

                msg.id = message_id
                stored.append(msg)
                seen_ids.add(message_id)

    def update_events(self, new_events: list[MindEvent], current_time: int) -> None:
        """Update event buffer with retention policy

        Events are distinct from observations - they're temporal occurrences that accumulate.
        Retention policy: Keep events that are:
        - Newer than EVENT_RETENTION_TIME_MINUTES game minutes, OR
        - Not yet marked as seen (will be marked after processing)
        Up to maximum of EVENT_BUFFER_MAX_SIZE most recent events

        Also extracts INTERACTION_BID_RECEIVED events and stores them separately
        in pending_incoming_bids for action generation.

        Args:
            new_events: New events from this decision cycle
            current_time: Current simulation time
        """
        for event in new_events:
            if event.event_type == MindEventType.INTERACTION_BID_RECEIVED:
                # Store incoming interaction bids for action generation
                bid_id = event.payload.get("bid_id")
                if bid_id:
                    self.pending_incoming_bids[bid_id] = event

            elif event.event_type == MindEventType.ERROR:
                # Log error events for debugging. Per-NPC line: attribute to the entity FK
                # so the sim /logs forwarder routes it to the NPC's Events tab.
                message = event.payload.get("message", "Unknown error")
                logger.warning(f"[{self.entity_id}] Received error event: {message}")

            elif event.event_type == MindEventType.INTERACTION_BID_CANCELED:
                # Remove canceled bid from pending list
                bid_id = event.payload.get("bid_id")
                if bid_id and bid_id in self.pending_incoming_bids:
                    del self.pending_incoming_bids[bid_id]
                    logger.debug(
                        f"[{self.entity_id}] Removed canceled bid {bid_id} from pending bids"
                    )

            elif event.event_type in (
                MindEventType.INTERACTION_FINISHED,
                MindEventType.INTERACTION_CANCELED,
            ):
                # Defer cleanup until build_pipeline_state snapshots this cycle.
                # A final conversation update and its finish event may share a
                # batch; deleting here would hide the closing turn from reflection.
                interaction_id = event.payload.get("interaction_id")
                if interaction_id:
                    self._finished_conversation_ids.add(interaction_id)

        self.event_buffer.extend(new_events)

        cutoff_time = current_time - EVENT_RETENTION_TIME_MINUTES
        retained = [e for e in self.event_buffer if e.timestamp > cutoff_time]

        if len(retained) > EVENT_BUFFER_MAX_SIZE:
            retained = sorted(retained, key=lambda e: e.timestamp, reverse=True)[
                :EVENT_BUFFER_MAX_SIZE
            ]

        self.event_buffer = retained

    def build_pipeline_state(self, obs: Observation) -> PipelineState:
        """The state one decision cycle runs on.

        Extracted out of ``server.py::decide_action`` so that anything measuring
        or exercising a cycle outside the MCP tool runs on the SAME construction
        production runs on. The predecessor measurement harness (NPC-1318) was
        described as mirroring this construction by hand; the mirror went stale
        silently, and its per-node table outlived the nodes it named.
        ``decide_action`` is a closure registered inside
        ``MCPServer._register_tools_and_resources``, so it is not importable —
        this method is the only shared surface a caller can reach.

        Callers own the per-cycle mutations that must precede it:
        ``update_conversations`` and ``update_events`` (which applies the
        retention policy that fills ``event_buffer``), plus stamping
        ``last_simulation_time``. Conversation cleanup is deferred until this
        method has copied the aggregate into the returned state, ensuring a
        closing turn remains visible during its finish cycle and disappears
        from the following cycle.

        Args:
            obs: This cycle's observation, already parsed and routed.

        Returns:
            A PipelineState carrying every input field the pipeline reads.
        """
        state = PipelineState(
            observation=obs,
            available_actions=obs.get_available_actions(
                pending_incoming_bids=self.pending_incoming_bids
            ),
            working_memory=self.working_memory,
            personality_traits=self.traits,
            personality_dimensions=self.personality_dimensions,
            conversation_histories={
                interaction_id: list(messages)
                for interaction_id, messages in self.conversation_histories.items()
            },
            recent_events=self.event_buffer,
            pending_incoming_bids=self.pending_incoming_bids,
        )
        for interaction_id in self._finished_conversation_ids:
            if self.conversation_histories.pop(interaction_id, None) is not None:
                logger.debug(
                    f"[{self.entity_id}] Cleaned up conversation history for {interaction_id}"
                )
        self._finished_conversation_ids.clear()
        return state
