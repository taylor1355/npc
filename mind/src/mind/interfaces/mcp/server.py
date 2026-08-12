"""MCP server for mind management"""

import json
import os
import uuid

from fastmcp import Context, FastMCP
from pydantic import ValidationError

from mind.cognitive_architecture.actions import ActionType
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
from mind.cognitive_architecture.nodes.memory_consolidation.node import MemoryConsolidationNode
from mind.cognitive_architecture.observations import (
    ConversationObservation,
    MindEvent,
    MindEventType,
    Observation,
)
from mind.cognitive_architecture.state import PipelineState
from mind.constants import DEFAULT_MEMORY_STORAGE_PATH
from mind.logging_config import get_logger

from .mind import Mind
from .models import (
    ConsolidationResponse,
    DecisionTelemetry,
    MindConfig,
    MindInfoResponse,
    MindStateResponse,
)

logger = get_logger()


## Wire-format version for decide_action responses. Bumped when the response
## shape changes in a way a client must know about. Emitted on BOTH branches so
## a client can distinguish "this server is older than telemetry" from "this
## server ran and reported nothing" -- two facts that would otherwise collapse
## into the same absent field.
PROTOCOL_VERSION = 1


def _error_response(request_id: str, error_message: str, details: str = None) -> dict:
    """Helper to construct error response dict.

    Carries no telemetry by design. An error means the cost of the attempt is
    genuinely unrecoverable here (retry exhaustion raises past any handler that
    can still reach PipelineState), and the client maps absent telemetry to
    "unknown", never to zero.
    """
    response = {
        "status": "error",
        "action": None,
        "error_message": error_message,
        "request_id": request_id,
        "protocol_version": PROTOCOL_VERSION,
    }
    if details:
        response["details"] = details
    return response


def _success_response(request_id: str, action: dict, telemetry: dict) -> dict:
    """Helper to construct success response dict"""
    return {
        "status": "success",
        "action": action,
        "error_message": None,
        "request_id": request_id,
        "protocol_version": PROTOCOL_VERSION,
        "telemetry": telemetry,
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
                logger.debug(
                    f"[{entity_id}] Skipping non-conversation interaction observation: {e}"
                )
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
            logger.debug(
                f"[{request_id}] [{entity_id}] Removed bid {bid_id} from pending bids after response"
            )

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

        logger.debug(
            f"[{request_id}] [{entity_id}] Batch rejected {len(bid_ids_to_remove)} bids: {bid_ids_to_remove}"
        )


class MCPServer:
    """MCP server for NPC minds"""

    def __init__(self, name="NPC Mind Server"):
        """Initialize the MCP server"""
        self.minds: dict[str, Mind] = {}  # Simple dict registry

        # mind_id (PK) -> the MindConfig that mind was CREATED with.
        #
        # Several MindConfig fields are client-settable per mind, so relink/forget
        # cannot default-construct a MindConfig(traits=[]) and expect it to describe
        # THIS mind (NPC-1023). Each defaulted field fails differently, and the
        # loudest is not the worst:
        #   - memory_storage_path decides WHERE the collection is looked for. Getting
        #     it wrong is at least self-announcing: relink reports "not_found".
        #   - embedding_model decides whether the vectors read back are comparable to
        #     the stored ones at all. Mind.reattach hands it to VectorDBMemory, which
        #     builds its own SentenceTransformer and embeds queries client-side, so a
        #     mind rehydrated under the default model queries a collection written by
        #     a different one - InvalidDimension when the widths differ, and silently
        #     meaningless nearest-neighbours when they happen to match. Nothing
        #     reports the second case, which makes it strictly worse than "not_found".
        #   - traits / personality_dimensions / llm_model decide who the rehydrated
        #     NPC actually is; defaulting them returns a personality-less stranger.
        #
        # This map deliberately OUTLIVES cleanup_mind: release retains the collection,
        # so the config must survive to make the later relink both addressable and
        # faithful. Only forget_mind drops the entry, because only forget destroys the
        # target.
        #
        # Growth: entries are added per created (or relinked) mind and removed only by
        # forget_mind, so the map grows monotonically in distinct client-supplied
        # mind_ids over the process lifetime. _config_to_record strips the seed payload,
        # which is what keeps an entry from pinning arbitrarily long prose - but an entry
        # is not small: traits and personality_dimensions are client-sized and retained.
        # See _config_to_record for the per-field accounting.
        #
        # Scope: process-local. It restores fidelity across an EVICTION; across a
        # server RESTART the map is empty and the client must supply the location
        # itself. See _config_for.
        self.mind_configs: dict[str, MindConfig] = {}

        # Create MCP server
        self.mcp = FastMCP(name)

        self._register_tools_and_resources()

    @staticmethod
    def _config_to_record(config: MindConfig) -> MindConfig:
        """Strip the seed payload from a config before recording it.

        The two stripped fields are dropped for different reasons, and only one of them
        is a no-op.

        initial_long_term_memories is genuinely ignored by Mind.reattach - re-seeding on
        every relink would duplicate the originals - so dropping it cannot change a
        rehydrated mind. It is also the field a client fills with arbitrarily long prose,
        so dropping it is what keeps each map entry practically bounded rather than
        pinning every NPC's seed text for the process lifetime, and makes "reattach never
        re-seeds" structural here instead of merely conventional.

        "Practically", not "entirely" - the stripping does not make an entry small. The
        client-sized fields are not just the two stripped here: traits and
        personality_dimensions are client-supplied collections too, and they are
        deliberately RETAINED, because a rehydrated mind must be the same character. An
        entry is therefore bounded by whatever personality the client sent, not by a
        constant. (initial_working_memory is unbounded in its own right whatever its
        declared fields, because WorkingMemory sets extra="allow".)

        initial_working_memory is different: Mind.reattach DOES read it (the
        `config.initial_working_memory or WorkingMemory()` line in mind.py), and there is
        no live mind in the branch that calls reattach to take it from. Dropping it is
        therefore a deliberate behavioral choice - a relinked mind starts from a blank
        WorkingMemory. Unlike traits or embedding_model, the creating snapshot is stale
        by relink time: the mind's working memory moved on while it was resident and was
        discarded at release, so replaying it would resurrect old state rather than
        restore fidelity. This also preserves the pre-NPC-1023 behavior exactly.
        """
        return config.model_copy(
            update={"initial_long_term_memories": [], "initial_working_memory": None}
        )

    def _config_for(self, mind_id: str, memory_storage_path: str | None = None) -> MindConfig:
        """Resolve the MindConfig to use for a non-resident mind.

        Precedence, strongest first:
          1. The config this server recorded for the mind. It wins even when the caller
             also supplies a path: a disagreeing caller path is a client error, not a
             relocation instruction, because this server has no move operation.
             The record has two provenances under one authority level. create_mind
             writes the config the mind was actually built with; relink_mind's restart
             branch writes back the case-2 reconstruction below (the caller's path plus
             defaults for everything else). Only the path affects addressing and it is
             the caller's own value either way, so the distinction is invisible today -
             but a recorded entry is not always a creating config, and the fidelity
             caveat on case 2 survives into any entry that came from it.
          2. A default config pointed at the caller-supplied memory_storage_path.
             This is the server-restart path: the map is empty, and the caller is the
             only remaining witness to where the collection lives.
          3. A fully default config, i.e. today's behavior for a caller that supplies
             nothing.

        Case 3 stays a genuine gap rather than a safe default: a mind created on a
        custom path with no path supplied here is unreachable, and relink_mind reports
        "not_found" honestly instead of mis-probing.

        Case 2 closes addressability across a restart but NOT fidelity: only the path
        crosses the process boundary, so a mind whose creating config set a non-default
        embedding_model, llm_model, traits, or personality_dimensions is rehydrated
        with the defaults for those. For embedding_model that is the silent-mismatch
        failure described on self.mind_configs, so a client that sets a non-default
        embedding_model must not rely on restart-relink to recover the mind.
        """
        recorded = self.mind_configs.get(mind_id)
        if recorded is not None:
            # Compare normalized, so "./db", "db" and "db/" - the same directory spelled
            # three ways - do not trip the warning. A diagnostic that cries wolf on a
            # caller who agrees with the record gets filtered out by that client before
            # the stale-path case it exists for ever arrives, which is worse than no
            # warning at all.
            #
            # normpath is purely textual, and deliberately so: it collapses "." and
            # redundant separators without touching the filesystem. It therefore does
            # NOT equate a symlink with its target, does NOT account for
            # case-insensitive filesystems, and does NOT equate a relative spelling with
            # an absolute one. Those still produce a false warning; realpath would fix
            # the first two at the cost of a filesystem call on every resolve, including
            # for paths that may not exist.
            if memory_storage_path and os.path.normpath(memory_storage_path) != os.path.normpath(
                recorded.memory_storage_path
            ):
                # Discarding the caller's path silently would let a destructive
                # forget_mind report success over a location the caller never named.
                # The record still wins - this is diagnosable, not negotiable.
                logger.warning(
                    f"Ignoring memory_storage_path={memory_storage_path!r} for {mind_id}: "
                    f"recorded config says {recorded.memory_storage_path!r}"
                )
            return recorded

        return MindConfig(
            traits=[],
            memory_storage_path=memory_storage_path or DEFAULT_MEMORY_STORAGE_PATH,
        )

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
            # Remember how this mind was built - not just where it lives - so a later
            # relink/forget can address it and rehydrate it faithfully once it is no
            # longer resident (NPC-1023).
            self.mind_configs[mind_id] = self._config_to_record(config)

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
            (the routed mind drives that entity); a divergence is logged (both ids) and
            the request is rejected, since a decision on a misrouted observation would
            be garbage.

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
                        request_id, f"Invalid observation format: {str(e)}", details=str(e)
                    )

                # Deserialize and validate events if provided
                mind_events = []
                if events is not None:
                    try:
                        mind_events = [MindEvent.model_validate(e) for e in events]
                    except ValidationError as e:
                        logger.exception(f"[{request_id}] Event validation failed for {mind_id}")
                        return _error_response(
                            request_id, f"Invalid event format: {str(e)}", details=str(e)
                        )

                # Defensive misrouting check: mind_id (PK) routes the request while the
                # observation carries its own entity_id (FK). In correct operation these
                # agree (the mind drives that entity); a divergence means the observation
                # was routed to the wrong mind. Fail loud at the boundary - log both ids
                # (so misrouting stays diagnosable) and reject, since a decision computed
                # on a misrouted observation would be garbage.
                if obs.entity_id != mind.entity_id:
                    logger.warning(
                        f"[{request_id}] entity_id mismatch for mind {mind_id}: "
                        f"observation entity_id={obs.entity_id} but mind entity_id={mind.entity_id} "
                        f"(misrouting — rejecting the request)"
                    )
                    return _error_response(
                        request_id,
                        f"entity_id mismatch: observation '{obs.entity_id}' "
                        f"but mind drives '{mind.entity_id}'",
                    )

                # Extract conversation observations from INTERACTION_OBSERVATION events.
                # Pass the entity FK so per-NPC log lines attribute to the NPC's Events tab.
                conversation_obs = _extract_conversation_observations(mind_events, mind.entity_id)
                mind.update_conversations(conversation_obs)
                mind.update_events(mind_events, obs.current_simulation_time)

                state = PipelineState(
                    observation=obs,
                    available_actions=obs.get_available_actions(
                        pending_incoming_bids=mind.pending_incoming_bids
                    ),
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
                telemetry = DecisionTelemetry.from_pipeline_state(result, mind.llm_model)
                return _success_response(
                    request_id, result.chosen_action.model_dump(), telemetry.model_dump()
                )

            except ValidationError as e:
                logger.warning(
                    f"[{request_id}] Validation failed in decide_action for {mind_id}: {str(e)}"
                )
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
            """Release a mind from memory while RETAINING its persisted collection.

            This drops the in-memory Mind instance (frees its pipeline/working state)
            but deliberately does NOT delete the ChromaDB collection. The retained
            collection is what lets a later relink_mind re-attach via Mind.reattach
            and recover the mind's long-term memory. To actually erase memory, use
            forget_mind. The "released" status reflects this retain-on-release
            contract (the mind is released, its memory persists).

            Args:
                mind_id: Mind to release
            """
            # The mind is still registered here, so its entity_id (FK) is available;
            # surface it in the response so release is symmetric with create_mind.
            # (The Godot client ignores this optional field, so this is non-breaking.)
            entity_id = None
            if mind_id in self.minds:
                entity_id = self.minds[mind_id].entity_id
                del self.minds[mind_id]

            return MindInfoResponse(status="released", mind_id=mind_id, entity_id=entity_id)

        @self.mcp.tool()
        async def relink_mind(
            mind_id: str,
            entity_id: str,
            memory_storage_path: str | None = None,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Re-bind a mind to a (possibly new) driven entity, rehydrating if needed.

            Three paths:
            - Mind still resident in self.minds: rebind its entity_id (FK) in place
              and report "relinked". No collection work - the live mind already holds
              its memory.
            - Mind released but its collection retained (cleanup_mind kept it): if
              collection_exists, rehydrate via Mind.reattach (no re-seeding), register
              it under the PK, and report "relinked".
            - Neither resident nor a retained collection: report "not_found".

            Args:
                mind_id: The mind's own identifier (PK) - keys self.minds and the
                    retained collection.
                entity_id: The simulation entity this mind should now drive (FK).
                memory_storage_path: Where to look for the retained collection when
                    this server has no record of the mind - i.e. after a restart.
                    Optional, and ignored when a recorded config exists (that config
                    is ground truth); omitting it preserves the previous behavior of
                    probing the default path, so an older client stays compatible.
                    Supplying it restores addressability but not the rest of the
                    creating config - see _config_for.
            """
            # Resident: rebind the FK in place. mind_id (PK) is the stable identity;
            # the driven entity can change across relinks.
            if mind_id in self.minds:
                self.minds[mind_id].entity_id = entity_id
                return MindInfoResponse(status="relinked", mind_id=mind_id, entity_id=entity_id)

            # Not resident: rehydrate from a retained collection if one survives.
            # The config must be the one this mind was CREATED with, not a freshly
            # defaulted one (NPC-1023). It decides both where to probe and how to
            # rebuild the mind - most sharply embedding_model, which must match the
            # model that wrote the stored vectors. See self.mind_configs.
            config = self._config_for(mind_id, memory_storage_path)
            if VectorDBMemory.collection_exists(config.memory_storage_path, f"mind_{mind_id}"):
                mind = Mind.reattach(mind_id, entity_id, config)
                self.minds[mind_id] = mind
                self.mind_configs[mind_id] = self._config_to_record(config)
                return MindInfoResponse(status="relinked", mind_id=mind_id, entity_id=entity_id)

            return MindInfoResponse(status="not_found", mind_id=mind_id, entity_id=entity_id)

        @self.mcp.tool()
        async def forget_mind(
            mind_id: str,
            memory_storage_path: str | None = None,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Permanently erase a mind's memory and drop it from the registry.

            The destructive counterpart to cleanup_mind: it deletes the persisted
            ChromaDB collection outright (not VectorDBMemory.clear, which would
            recreate an empty collection), so collection_exists is False afterward
            and a subsequent relink_mind reports "not_found". Also drops the
            in-memory Mind if still resident.

            Returns "forgotten" only when something was actually erased - a resident
            mind, a retained collection, or both. When neither is found the status is
            "not_found", matching relink_mind. "forgotten" is a claim that memory was
            destroyed, so it must never be returned over a collection that survived:
            telling the caller a mind is gone when it is not is worse than an honest
            "not_found" (NPC-1023).

            Args:
                mind_id: Mind to forget
                memory_storage_path: Where to look for the collection when this server
                    has no record of the mind - i.e. after a restart. Optional and
                    ignored when a recorded config exists; omitting it preserves the
                    previous default-path behavior, so an older client stays
                    compatible. Without it, a restarted server cannot erase a mind
                    that lived on a custom path and reports "not_found" rather than
                    claiming a deletion it did not perform.
            """
            entity_id = None
            # Erase where this mind actually lives, not where the default config
            # points - see _config_for (NPC-1023).
            storage_path = self._config_for(mind_id, memory_storage_path).memory_storage_path
            collection_name = f"mind_{mind_id}"
            erased = False

            # Resolve the FK (if resident) and drop the in-memory instance. The live
            # store's client is reused for the delete when resident; otherwise open a
            # client just to delete the retained collection.
            if mind_id in self.minds:
                mind = self.minds[mind_id]
                entity_id = mind.entity_id
                # Delete-if-exists, matching the non-resident branch below. An
                # unguarded raise here would skip the registry drop on the next
                # line, leaving a mind resident that the caller was told to forget
                # - the collection being already gone is the desired end state, not
                # a failure. The store's collection is keyed by the same mind PK as
                # collection_name above (Mind.from_config / Mind.reattach).
                mind.memory_store.drop_collection()
                del self.minds[mind_id]
                # A resident mind is something to erase in its own right: dropping the
                # live instance is a real effect even if its collection was already
                # gone (the tolerates-an-already-deleted-collection path).
                erased = True
            elif VectorDBMemory.collection_exists(storage_path, collection_name):
                # Non-resident: delete via a bare client (no encoder load, no
                # get_or_create_collection that would recreate the collection first).
                VectorDBMemory.delete_collection(storage_path, collection_name)
                erased = True

            if not erased:
                return MindInfoResponse(status="not_found", mind_id=mind_id, entity_id=entity_id)

            # Only forget drops the recorded config - cleanup_mind must keep it so the
            # retained collection stays addressable by a later relink.
            self.mind_configs.pop(mind_id, None)
            return MindInfoResponse(status="forgotten", mind_id=mind_id, entity_id=entity_id)

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
