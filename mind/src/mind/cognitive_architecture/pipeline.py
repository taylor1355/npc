"""LangGraph cognitive pipeline implementation"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from .memory.vector_db_memory import VectorDBMemory
from .nodes.action_selection.node import ActionSelectionNode
from .nodes.cognitive_update.node import CognitiveUpdateNode
from .nodes.memory_query.node import MemoryQueryNode
from .nodes.memory_retrieval.node import MemoryRetrievalNode
from .state import PipelineState


class CognitivePipeline:
    """Orchestrates the cognitive processing pipeline using LangGraph"""

    def __init__(self, llm: BaseChatModel, memory_store: VectorDBMemory):
        self.llm = llm
        self.memory_store = memory_store

        # Initialize nodes
        self.memory_query_node = MemoryQueryNode(llm)
        self.memory_retrieval_node = MemoryRetrievalNode(memory_store)
        self.cognitive_update_node = CognitiveUpdateNode(llm)
        self.action_selection_node = ActionSelectionNode(llm)

        # Build the graph
        self.graph = self._build_graph()
        self.chain = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph processing graph"""

        # Create graph with state type
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("memory_query", self.memory_query_node.process)
        workflow.add_node("memory_retrieval", self.memory_retrieval_node.process)
        workflow.add_node("cognitive_update", self.cognitive_update_node.process)
        workflow.add_node("action_selection", self.action_selection_node.process)

        # Define edges
        workflow.set_entry_point("memory_query")
        workflow.add_edge("memory_query", "memory_retrieval")
        workflow.add_edge("memory_retrieval", "cognitive_update")
        workflow.add_edge("cognitive_update", "action_selection")
        workflow.add_edge("action_selection", END)

        return workflow

    async def process(self, state: PipelineState) -> PipelineState:
        """Process an observation through the cognitive pipeline"""
        result_dict = await self.chain.ainvoke(state)
        # Convert LangGraph's AddableValuesDict back to our Pydantic model
        return PipelineState(**result_dict)
