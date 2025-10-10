"""Mind runtime state and behavior"""

from dataclasses import dataclass, field
from typing import Self

from mind.cognitive_architecture.models import ConversationMessage
from mind.cognitive_architecture.nodes.cognitive_update.models import NewMemory, WorkingMemory
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
from mind.cognitive_architecture.pipeline import CognitivePipeline
from mind.apis.langchain_llm import get_llm


@dataclass
class Mind:
    """Runtime state for a single mind - encapsulates all mind behavior"""
    mind_id: str
    entity_id: str  # Mind's entity ID in simulation
    traits: list[str]
    pipeline: CognitivePipeline
    memory_store: VectorDBMemory
    working_memory: WorkingMemory
    daily_memories: list[NewMemory] = field(default_factory=list)

    # Conversation history aggregation (keyed by interaction_id)
    conversation_histories: dict[str, list[ConversationMessage]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, mind_id: str, config) -> Self:
        """Create a Mind instance from configuration

        Args:
            mind_id: Unique identifier for the mind
            config: MindConfig with LLM, memory, and initial state settings

        Returns:
            Initialized Mind instance
        """
        # Initialize LLM from config
        llm = get_llm(config.llm_model)

        # Initialize memory store with configured collection name
        memory_store = VectorDBMemory(
            collection_name=f"mind_{mind_id}",
            embedding_model=config.embedding_model,
            storage_path=config.memory_storage_path
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
            entity_id=config.entity_id,
            traits=config.traits,
            pipeline=pipeline,
            memory_store=memory_store,
            working_memory=working_memory,
        )

    def update_conversations(self, conversations: list) -> None:
        """Aggregate conversation updates into full history

        Args:
            conversations: List of ConversationObservation objects
        """
        for conv_obs in conversations:
            interaction_id = conv_obs.interaction_id

            # Initialize if new conversation
            if interaction_id not in self.conversation_histories:
                self.conversation_histories[interaction_id] = []

            # Append new messages (avoid duplicates by checking timestamps)
            existing_timestamps = {
                msg.timestamp
                for msg in self.conversation_histories[interaction_id]
                if msg.timestamp is not None
            }

            for msg in conv_obs.conversation_history:
                if msg.timestamp is None or msg.timestamp not in existing_timestamps:
                    self.conversation_histories[interaction_id].append(msg)
