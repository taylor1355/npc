"""Mind runtime state and behavior"""

from dataclasses import dataclass, field
from typing import Self

from mind.apis.langchain_llm import get_llm
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
from mind.cognitive_architecture.observations import ConversationMessage, MindEvent, MindEventType
from mind.cognitive_architecture.pipeline import CognitivePipeline
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

    event_buffer: list[MindEvent] = field(default_factory=list)

    # Pending incoming interaction bids (keyed by bid_id from payload)
    pending_incoming_bids: dict[str, MindEvent] = field(default_factory=dict)

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

        # Seed initial long-term memories
        for memory_content in config.initial_long_term_memories:
            memory_store.add_memory(content=memory_content, importance=5.0)

        # Initialize pipeline
        pipeline = CognitivePipeline(llm=llm, memory_store=memory_store)

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

        # Initialize pipeline
        pipeline = CognitivePipeline(llm=llm, memory_store=memory_store)

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

    @staticmethod
    def _composite_message_key(msg: ConversationMessage) -> tuple:
        """Identity for a message carrying no id, from the fields the wire always has.

        ``speaker_id`` is part of the key because a timestamp alone is not an
        identity: the simulation truncates game time to whole minutes, and two
        different speakers routinely land in the same minute (a participant's
        message and the system's "message limit reached" notice are appended in
        the same tick, with no game time between them).

        ``timestamp`` may be ``None``; that is a legitimate tuple element rather
        than a reason to skip dedup, which is what makes an unstamped message
        deduplicate like any other.
        """
        return (msg.timestamp, msg.speaker_id, msg.message)

    def update_conversations(self, conversations: list) -> None:
        """Aggregate conversation updates into full history.

        Observations arrive as overlapping rolling windows, so the same message
        is re-sent on many cycles and must be stored exactly once.

        Dedup has **two branches, and the composite key is not a migration
        step** — it is the permanent fallback for any message whose producer
        sent no id (see ``ConversationMessage.id``: an older simulation is a
        shape this server must accept forever, because the two halves are
        deployed independently).

        1. **By id** when the message carries one — durable across re-sends and
           immune to two speakers sharing a truncated timestamp.
        2. **By composite key** ``(timestamp, speaker_id, message)`` otherwise.

        Both indices are updated *inside* the loop, so a batch containing the
        same message twice stores it once. Building them once up front was the
        defect that let intra-batch duplicates through.

        Args:
            conversations: List of ConversationObservation objects
        """
        for conv_obs in conversations:
            interaction_id = conv_obs.interaction_id

            # Initialize if new conversation
            if interaction_id not in self.conversation_histories:
                self.conversation_histories[interaction_id] = []

            stored = self.conversation_histories[interaction_id]

            by_id = {msg.id: msg for msg in stored if msg.id}
            # Built from ALL stored messages, not just id-less ones, so a message
            # stored WITH an id is still recognised when it re-arrives WITHOUT one
            # (the simulation downgraded between cycles). First writer wins: on a
            # composite collision the index points at the earliest copy, which is
            # the one an upgrade should backfill onto.
            by_composite: dict[tuple, ConversationMessage] = {}
            for msg in stored:
                by_composite.setdefault(self._composite_message_key(msg), msg)

            for msg in conv_obs.conversation_history:
                incoming_id = msg.id
                if incoming_id is not None and not incoming_id.strip():
                    # An EMPTY id is a producer bug, categorically different from
                    # an ABSENT one: the simulation mints in the constructor, so
                    # a blank means the mint path was bypassed. Never treat "" as
                    # an identity (every such message would collide with every
                    # other) — say so and fall through to the composite branch,
                    # which still deduplicates it correctly.
                    logger.warning(
                        f"[{self.entity_id}] Conversation message arrived with an EMPTY id "
                        f"in {interaction_id} (speaker={msg.speaker_id!r}) — the producer's "
                        f"mint path was bypassed. Falling back to the composite key."
                    )
                    incoming_id = None

                if incoming_id:
                    if incoming_id in by_id:
                        continue

                    key = self._composite_message_key(msg)
                    prior = by_composite.get(key)
                    if prior is not None and not prior.id:
                        # Upgrade path: we already hold this message from a cycle
                        # that carried no id. Backfill the id onto the copy we
                        # hold rather than appending a second copy of one message.
                        prior.id = incoming_id
                        by_id[incoming_id] = prior
                        continue
                    if prior is not None and prior.id and prior.id != incoming_id:
                        # Distinct ids mean the producer considers these distinct
                        # messages, so keep both — the composite key is the weaker
                        # identity and must not overrule an explicit one.
                        logger.warning(
                            f"[{self.entity_id}] Two messages in {interaction_id} share a "
                            f"composite key but carry distinct ids "
                            f"({prior.id!r} vs {incoming_id!r}) — keeping both."
                        )

                    stored.append(msg)
                    by_id[incoming_id] = msg
                    by_composite.setdefault(key, msg)
                    continue

                key = self._composite_message_key(msg)
                if key in by_composite:
                    continue
                stored.append(msg)
                by_composite[key] = msg

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
                # Clean up conversation history for ended interactions
                interaction_id = event.payload.get("interaction_id")
                if interaction_id and interaction_id in self.conversation_histories:
                    del self.conversation_histories[interaction_id]
                    logger.debug(
                        f"[{self.entity_id}] Cleaned up conversation history for {interaction_id}"
                    )

        self.event_buffer.extend(new_events)

        cutoff_time = current_time - EVENT_RETENTION_TIME_MINUTES
        retained = [e for e in self.event_buffer if e.timestamp > cutoff_time]

        if len(retained) > EVENT_BUFFER_MAX_SIZE:
            retained = sorted(retained, key=lambda e: e.timestamp, reverse=True)[
                :EVENT_BUFFER_MAX_SIZE
            ]

        self.event_buffer = retained
