"""MCP server for mind management"""

import json
from typing import Dict

from fastmcp import FastMCP, Context

from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.nodes.memory_consolidation.node import MemoryConsolidationNode
from mind.cognitive_architecture.models import Observation

from .mind import Mind
from .models import (
    MindConfig,
    SimulationRequest,
    MindResponse,
    MindStateResponse,
    ConsolidationResponse,
    MindInfoResponse,
)


class MCPServer:
    """MCP server for NPC minds"""

    def __init__(self, name="NPC Mind Server"):
        """Initialize the MCP server"""
        self.minds: Dict[str, Mind] = {}  # Simple dict registry

        # Create MCP server
        self.mcp = FastMCP(name)

        self._register_tools_and_resources()

    def _register_tools_and_resources(self):
        """Register all tools and resources with MCP"""

        # === Tools ===

        @self.mcp.tool()
        async def create_mind(
            mind_id: str,
            config: MindConfig,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Create a new NPC mind

            Args:
                mind_id: Unique identifier for the mind (usually matches entity_id)
                config: Configuration with entity_id, traits, LLM settings, memory settings, initial state
            """
            mind = Mind.from_config(mind_id, config)
            self.minds[mind_id] = mind

            return MindInfoResponse(
                status="created",
                mind_id=mind_id
            )

        @self.mcp.tool()
        async def decide_action(
            request: SimulationRequest,
            ctx: Context = None,
        ) -> MindResponse:
            """Process observation from simulation and decide on an action

            Args:
                request: SimulationRequest with mind_id and structured observation
            """
            # Get mind
            if request.mind_id not in self.minds:
                return MindResponse(
                    status="error",
                    error_message=f"Mind {request.mind_id} not found"
                )

            mind = self.minds[request.mind_id]

            # Update conversation histories
            mind.update_conversations(request.observation.conversations)

            # Build pipeline state
            state = PipelineState(
                observation=request.observation,
                working_memory=mind.working_memory,
                personality_traits=mind.traits,
                conversation_histories=mind.conversation_histories,
            )

            # Run pipeline
            result = await mind.pipeline.process(state)

            # Update mind state
            mind.working_memory = result.working_memory
            mind.daily_memories.extend(result.daily_memories)

            return MindResponse(
                status="success",
                action=result.chosen_action
            )

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
                return ConsolidationResponse(
                    status="error",
                    consolidated_count=0
                )

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

            # Clear daily buffer
            count = len(mind.daily_memories)
            mind.daily_memories.clear()

            return ConsolidationResponse(
                status="success",
                consolidated_count=count
            )

        @self.mcp.tool()
        async def cleanup_mind(
            mind_id: str,
            ctx: Context = None,
        ) -> MindInfoResponse:
            """Gracefully cleanup and remove a mind

            Args:
                mind_id: Mind to remove
            """
            if mind_id in self.minds:
                del self.minds[mind_id]

            return MindInfoResponse(
                status="removed",
                mind_id=mind_id
            )

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
                active_conversations=list(mind.conversation_histories.keys())
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
                indent=2
            )
