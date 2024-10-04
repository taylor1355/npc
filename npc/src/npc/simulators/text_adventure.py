from dataclasses import dataclass
from typing import Dict, List, Optional
from npc.llm_response_generator import LLMResponseGenerator
from npc.prompts.common import TagPattern
from npc.prompts.text_adventure.story_concept_template import prompt as story_concept_prompt
from npc.prompts.text_adventure.story_section_template import prompt as story_section_prompt


@dataclass
class State:
    observation: str
    available_actions: Dict[int, str]


class Action:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __str__(self):
        return f"*{self.name}*. {self.description}"


class StoryNode:
    def __init__(self, story_section: str, actions: List[Action], parent: Optional["StoryNode"] = None):
        self.story_section = story_section
        self.actions = actions
        self.parent = parent
        self.children: Dict[int, StoryNode] = {}

    @staticmethod
    def from_llm_response(llm_response: dict, parent: Optional["StoryNode"] = None) -> "StoryNode":
        text = llm_response["next_section"]
        name_pattern = TagPattern("name")
        description_pattern = TagPattern("description")
        actions = [
            Action(name_pattern.extract_from(action), description_pattern.extract_from(action))
            for action in llm_response["actions"]
        ]
        return StoryNode(text, actions, parent)


# TODO: create abstract base class for simulators
class TextAdventureSimulator:
    def __init__(self, llm):
        self.llm = llm
        self.story_guide = self._generate_story_guide()
        self.root = self._generate_initial_node()
        self.current_node = self.root

    def _generate_story_guide(self) -> str:
        story_concept_generator = LLMResponseGenerator(story_concept_prompt, self.llm)
        response = story_concept_generator.generate_response()
        return response["guide"]

    def _generate_initial_node(self) -> StoryNode:
        story_section_generator = LLMResponseGenerator(story_section_prompt, self.llm)
        response = story_section_generator.generate_response(
            guide=self.story_guide,
            previous_sections="",
            previous_action=""
        )
        return StoryNode.from_llm_response(response)

    @property
    def state(self) -> State:
        return State(
            observation=self.current_node.story_section,
            available_actions={i+1: str(action) for i, action in enumerate(self.current_node.actions)}
        )

    def take_action(self, action_index: int) -> State:
        if action_index not in self.state.available_actions:
            raise ValueError("Invalid action index")
        
        if action_index not in self.current_node.children:
            self.current_node.children[action_index] = self._generate_next_node(action_index)

        self.current_node = self.current_node.children[action_index]
        return self.state

    def _generate_next_node(self, action_index: int) -> StoryNode:
        previous_sections = self._get_story_so_far()
        previous_action = self.current_node.actions[action_index - 1] # TODO: fix off by 1 indexing
        
        story_section_generator = LLMResponseGenerator(story_section_prompt, self.llm)
        response = story_section_generator.generate_response(
            guide=self.story_guide,
            previous_sections=previous_sections,
            previous_action=f"{previous_action.name}: {previous_action.description}"
        )
        
        if response.get("epilogue"):
            return StoryNode(response["epilogue"], [], parent=self.current_node)
        
        return StoryNode.from_llm_response(response, self.current_node)

    def _get_story_so_far(self) -> str:
        story = []
        node = self.current_node
        while node:
            story.append(node.story_section)
            node = node.parent
        return "\n\n".join(reversed(story))

    def is_story_ended(self) -> bool:
        return len(self.current_node.actions) == 0