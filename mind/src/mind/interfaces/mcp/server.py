"""MCP server for mind management"""

import json
import logging
import uuid

from fastmcp import Context, FastMCP
from pydantic import ValidationError

from mind.cognitive_architecture.actions import ActionType
from mind.cognitive_architecture.observations import (
    ConversationObservation,
    MindEvent,
    MindEventType,
    Observation,
)
from mind.cognitive_architecture.nodes.memory_consolidation.node import MemoryConsolidationNode
from mind.cognitive_architecture.state import PipelineState
from mind.logging_config import get_logger

from .mind import Mind
from .models import (
    ConsolidationResponse,
    MindConfig,
    MindInfoResponse,
    MindStateResponse,
)

logger = get_logger()


def _error_response(request_id: str, error_message: str, details: str = None) -> dict:
    """Helper to construct error response dict"""
    response = {
        "status": "error",
        "action": None,
        "error_message": error_message,
        "request_id": request_id,
    }
    if details:
        response["details"] = details
    return response


def _success_response(request_id: str, action: dict) -> dict:
    """Helper to construct success response dict"""
    return {
        "status": "success",
        "action": action,
        "error_message": None,
        "request_id": request_id,
    }


def _extract_conversation_observations(
    events: list[MindEvent], entity_id: str
) -> list[ConversationObservation]:
    """Extract conversation observations from INTERACTION_OBSERVATION events.

    Args:
        events: List of MindEvent objects
        entity_id: Driven entity FK these events belong to. This is a per-NPC log line,
            so it carries entity_id (not the mind PK): the sim /logs forwarder
            regex-attributes the tag to the NPC's Events tab.

    Returns:
        List of ConversationObservation objects parsed from interaction observation events
    """
    conversations = []
    for event in events:
        if event.event_type == MindEventType.INTERACTION_OBSERVATION:
            try:
                # Parse the payload as ConversationObservation
                conv_obs = ConversationObservation.model_validate(event.payload)
                conversations.append(conv_obs)
            except ValidationError as e:
                # Not a conversation observation or malformed - skip it
                logger.debug(f"[{entity_id}] Skipping non-conversation interaction observation: {e}")
                continue
    return conversations


def _cleanup_responded_bids(action, pending_bids: dict, request_id: str, entity_id: str) -> None:
    """Remove bids from pending list after responding to them.

    Args:
        action: The chosen action (Action model)
        pending_bids: Dict of pending incoming bids (modified in place)
        request_id: Server-routing correlation id (not NPC-attributed)
        entity_id: Driven entity FK the bids belong to. These are per-NPC log lines,
            so they carry entity_id (not the mind PK): the sim /logs forwarder
            regex-attributes the tag to the NPC's Events tab.
    """
    if not action:
        return

    if action.action == ActionType.RESPOND_TO_INTERACTION_BID:
        # Single bid response
        bid_id = action.parameters.get("bid_id")
        if bid_id and bid_id in pending_bids:
            pending_bids.pop(bid_id)
            logger.debug(f"[{request_id}] [{entity_id}] Removed bid {bid_id} from pending bids after response")

    elif action.action == ActionType.BATCH_REJECT_INTERACTION_BIDS:
        # Batch bid rejection
        ids_param = action.parameters.get("ids")
        if not ids_param:
            return

        bid_ids_to_remove = []

        if ids_param == "*":
            # Reject all pending bids
            bid_ids_to_remove = list(pending_bids.keys())
        elif isinstance(ids_param, list):
            # Check if these are bid IDs or entity IDs
            for item in ids_param:
                if item in pending_bids:
                    # Direct bid ID
                    bid_ids_to_remove.append(item)
                else:
                    # Entity ID - find all bids from this entity
                    for bid_id, event in pending_bids.items():
                        if event.payload.get("bidder_id") == item:
                            bid_ids_to_remove.append(bid_id)

        # Remove the bids
        for bid_id in bid_ids_to_remove:
            pending_bids.pop(bid_id, None)

        logger.debug(f"[{request_id}] [{entity_id}] Batch rejected {len(bid_ids_to_remove)} bids: {bid_ids_to_remove}")


class MCPServer:
    """MCP server for NPC minds"""

    def __init__(self, name="NPC Mind Server"):
        """Initialize the MCP server"""
        self.minds: dict[str, Mind] = {}  # Simple dict registry

        # Create MCP server
        self.mcp = FastMCP(name)

        self._register_tools_and_resources()

    def _register_tools_and_resources(self):
        """Register all tools and resources with MCP"""

        # === Tools ===

        @self.mcp.tool()
        async def create_mind(
            mind_id: str,
            entity_id: str,
            config: MindConfig,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Create a new NPC mind

            mind_id (PK) and entity_id (FK) are deliberately distinct first-class ids:
            the mind owns its memory under the PK; the FK names the simulation entity
            the mind drives and is what per-NPC logs attribute to.

            Args:
                mind_id: The mind's own identifier (PK). Keys self.minds and the
                    memory collection.
                entity_id: The simulation entity this mind drives (FK).
                config: Cognitive configuration - traits, LLM settings, memory
                    settings, personality dimensions, initial state.
            """
            mind = Mind.from_config(mind_id, entity_id, config)
            self.minds[mind_id] = mind

            return MindInfoResponse(status="created", mind_id=mind_id, entity_id=entity_id)

        @self.mcp.tool()
        async def decide_action(
            mind_id: str,
            observation: dict,
            events: list = None,
            ctx: Context = None,
        ) -> dict:
            """Process observation from simulation and decide on an action

            mind_id is the routing primary key: it selects which mind in self.minds
            handles this request and keys that mind's memory collection. It is distinct
            from the entity_id foreign key carried inside the observation, which names
            the simulation entity the mind drives. In correct operation the two agree
            (the routed mind drives that entity); a divergence is logged as a possible
            misrouting (see the mismatch warning below) but does not reject the request.

            Args:
                mind_id: Routing primary key (PK) selecting the mind; distinct from the
                    observation entity_id foreign key (FK).
                observation: Structured observation dict (will be validated to Observation
                    model); carries the entity_id FK the pipeline attributes work to.
                events: List of mind events

            Returns:
                dict with status, action, error_message, and request_id
            """
            request_id = str(uuid.uuid4())[:8]
            logger.debug(f"[{request_id}] decide_action called for mind_id={mind_id}")

            try:
                if mind_id not in self.minds:
                    logger.warning(f"[{request_id}] Mind {mind_id} not found")
                    return _error_response(request_id, f"Mind {mind_id} not found")

                mind = self.minds[mind_id]

                # Validate observation
                try:
                    obs = Observation.model_validate(observation)
                except ValidationError as e:
                    logger.exception(f"[{request_id}] Observation validation failed for {mind_id}")
                    return _error_response(
                        request_id,
                        f"Invalid observation format: {str(e)}",
                        details=str(e)
                    )

                # Deserialize and validate events if provided
                mind_events = []
                if events is not None:
                    try:
                        mind_events = [MindEvent.model_validate(e) for e in events]
                    except ValidationError as e:
                        logger.exception(f"[{request_id}] Event validation failed for {mind_id}")
                        return _error_response(
                            request_id,
                            f"Invalid event format: {str(e)}",
                            details=str(e)
                        )

                # Defensive misrouting check: mind_id (PK) routes the request while the
                # observation carries its own entity_id (FK). In correct operation these
                # agree (the mind drives that entity); a divergence means the observation
                # was routed to the wrong mind. Warn (with both ids) so it is diagnosable,
                # but do not reject - the pipeline keeps using the observation's entity_id.
                if obs.entity_id != mind.entity_id:
                    logger.warning(
                        f"[{request_id}] entity_id mismatch for mind {mind_id}: "
                        f"observation entity_id={obs.entity_id} but mind entity_id={mind.entity_id} "
                        f"(possible misrouting; using observation entity_id for the pipeline)"
                    )

                # Extract conversation observations from INTERACTION_OBSERVATION events.
                # Pass the entity FK so per-NPC log lines attribute to the NPC's Events tab.
                conversation_obs = _extract_conversation_observations(mind_events, mind.entity_id)
                mind.update_conversations(conversation_obs)
                mind.update_events(mind_events, obs.current_simulation_time)

                state = PipelineState(
                    observation=obs,
                    available_actions=obs.get_available_actions(pending_incoming_bids=mind.pending_incoming_bids),
                    working_memory=mind.working_memory,
                    personality_traits=mind.traits,
                    personality_dimensions=mind.personality_dimensions,
                    conversation_histories=mind.conversation_histories,
                    recent_events=mind.event_buffer,
                    pending_incoming_bids=mind.pending_incoming_bids,
                )

                # Run cognitive pipeline
                logger.debug(f"[{request_id}] Running cognitive pipeline for {mind_id}")
                result = await mind.pipeline.process(state)

                mind.working_memory = result.working_memory
                mind.daily_memories.extend(result.daily_memories)
                mind.event_buffer = result.recent_events

                # Clean up any bids that were responded to. Pass the entity FK so the
                # per-NPC bid-cleanup log lines attribute to the NPC's Events tab.
                _cleanup_responded_bids(
                    result.chosen_action, mind.pending_incoming_bids, request_id, mind.entity_id
                )

                if result.chosen_action is None:
                    logger.warning(f"[{request_id}] Pipeline returned no action for {mind_id}")
                    return _error_response(request_id, "Pipeline did not select an action")

                logger.info(
                    f"[{request_id}] Successfully processed decision for {mind_id}: {result.chosen_action.action}"
                )
                return _success_response(request_id, result.chosen_action.model_dump())

            except ValidationError as e:
                logger.warning(f"[{request_id}] Validation failed in decide_action for {mind_id}: {str(e)}")
                return _error_response(request_id, "Action validation failed", details=str(e))
            except Exception:
                logger.exception(f"[{request_id}] Unexpected error in decide_action for {mind_id}")
                return _error_response(request_id, "Unexpected server error")

        @self.mcp.tool()
        async def consolidate_memories(
            mind_id: str,
            ctx: Context = None,
        ) -> ConsolidationResponse:
            """Consolidate daily memories into long-term storage

            Args:
                mind_id: Mind to consolidate memories for
            """
            if mind_id not in self.minds:
                return ConsolidationResponse(status="error", consolidated_count=0)

            mind = self.minds[mind_id]

            # Create dummy state for consolidation
            # TODO: Track latest observation for better location/timestamp
            dummy_obs = Observation(
                entity_id=mind.entity_id,
                current_simulation_time=0,
            )
            dummy_state = PipelineState(
                observation=dummy_obs,
                daily_memories=mind.daily_memories,
            )

            # Run consolidation
            consolidation_node = MemoryConsolidationNode(mind.memory_store)
            await consolidation_node.process(dummy_state)

            # Clear daily buffer and return count
            count = len(mind.daily_memories)
            mind.daily_memories.clear()

            return ConsolidationResponse(status="success", consolidated_count=count)

        @self.mcp.tool()
        async def cleanup_mind(
            mind_id: str,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Gracefully cleanup and remove a mind

            Args:
                mind_id: Mind to remove
            """
            # The mind is still registered here, so its entity_id (FK) is available;
            # surface it in the response so cleanup is symmetric with create_mind.
            # (The Godot client ignores this optional field, so this is non-breaking.)
            entity_id = None
            if mind_id in self.minds:
                entity_id = self.minds[mind_id].entity_id
                del self.minds[mind_id]

            return MindInfoResponse(status="removed", mind_id=mind_id, entity_id=entity_id)

        # === Resources ===

        @self.mcp.resource("mind://{mind_id}/state")
        async def get_mind_state(mind_id: str) -> str:
            """Get the mind's complete mental state"""
            if mind_id not in self.minds:
                return json.dumps({"error": f"Mind {mind_id} not found"})

            mind = self.minds[mind_id]

            state_response = MindStateResponse(
                entity_id=mind.entity_id,
                traits=mind.traits,
                working_memory=mind.working_memory,
                daily_memories_count=len(mind.daily_memories),
                long_term_memory_count=mind.memory_store.collection.count(),
                active_conversations=list(mind.conversation_histories.keys()),
            )

            return state_response.model_dump_json(indent=2)

        @self.mcp.resource("mind://{mind_id}/working_memory")
        async def get_working_memory(mind_id: str) -> str:
            """Get mind's current working memory"""
            if mind_id not in self.minds:
                return json.dumps({"error": f"Mind {mind_id} not found"})

            mind = self.minds[mind_id]
            return mind.working_memory.model_dump_json(indent=2)

        @self.mcp.resource("mind://{mind_id}/daily_memories")
        async def get_daily_memories(mind_id: str) -> str:
            """Get mind's accumulated daily memories"""
            if mind_id not in self.minds:
                return json.dumps({"error": f"Mind {mind_id} not found"})

            mind = self.minds[mind_id]
            return json.dumps(
                [{"content": m.content, "importance": m.importance} for m in mind.daily_memories],
                indent=2,
            )
