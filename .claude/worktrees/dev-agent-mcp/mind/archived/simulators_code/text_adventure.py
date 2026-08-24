import ast
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from pprint import pformat

from mind.apis.llm_client import LLMFunction
from mind.prompts.prompt_common import TagPattern
from mind.prompts.text_adventure.story_concept_template import prompt as story_concept_prompt
from mind.prompts.text_adventure.story_outcomes_template import prompt as story_outcomes_prompt
from mind.prompts.text_adventure.story_section_template import prompt as story_section_prompt


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


def softmax(likelihoods, temperature=1.0):
    likelihoods = np.array(likelihoods)
    return np.exp(likelihoods / temperature) / np.sum(np.exp(likelihoods / temperature))


# TODO: keep track of time using a string representation. This should help with pacing
# TODO: update the guide if the story has diverged significantly. This should keep things less repetitive.
# Minimize output length by having the model give a find and replace command to update the guide.
class StoryNode:
    def __init__(
            self,
            story_guide: str,
            llm,
            previous_action: str = "",
            parent: Optional["StoryNode"] = None
        ):

        self.story_guide = story_guide
        self.llm = llm
        self.previous_action = previous_action
        self.story_section = ""
        self.parent = parent
        self.children: Dict[int, StoryNode] = {}

        story_so_far = self.story_so_far()
        sampled_outcome = ""
        if previous_action:
            # Generate outcomes
            story_outcomes_generator = LLMFunction(story_outcomes_prompt, self.llm)
            self.story_outcomes_response = story_outcomes_generator.generate(
                guide=self.story_guide,
                previous_sections=story_so_far,
                previous_action=self.previous_action
            )
            self.outcomes = [{
                "description": description,
                "likelihood": float(likelihood.strip()),
                "ends_story": ast.literal_eval(ends_story)
            } for description, likelihood, ends_story in zip(
                self.story_outcomes_response["outcomes"],
                self.story_outcomes_response["likelihoods"],
                self.story_outcomes_response["ends_stories"]
            )]

            # Sample outcome based on likelihood
            outcome_likelihoods = [outcome["likelihood"] for outcome in self.outcomes]
            sampling_temperature = 3.0
            sampled_outcome_index = np.random.choice(
                len(self.outcomes),
                p=softmax(outcome_likelihoods, temperature=sampling_temperature)
            )
            sampled_outcome = self.outcomes[sampled_outcome_index]

        # Generate next section based on the sampled outcome
        story_section_generator = LLMFunction(story_section_prompt, self.llm)
        self.story_section_response = story_section_generator.generate(
            guide=self.story_guide,
            previous_sections=story_so_far,
            previous_action=self.previous_action,
            previous_outcome=sampled_outcome["description"] if sampled_outcome else ""
        )
        self.story_section = self.story_section_response["next_section"]
        self.actions = [
            Action(
                TagPattern("name").extract_from(action),
                TagPattern("description").extract_from(action)
            )
            for action in self.story_section_response["actions"]
        ]

        # TODO: use a logger
        if True:
            print(f"Story so far: {story_so_far}")
            if self.previous_action:
                # print caluculated outcome probabilities (outcome name: probability)
                outcome_probabilities_map = dict(zip([outcome['description'] for outcome in self.outcomes], softmax(outcome_likelihoods, temperature=sampling_temperature)))
                print(f"Outcome probabilities: {pformat(outcome_probabilities_map, width=120)}")
                print(f"Chosen outcome: {sampled_outcome}")
            print(f"Next story section: {self.story_section}")
            print(f"Actions: {[str(action) for action in self.actions]}")

    # TODO: use summarization to avoid blowing up the LLM context
    def story_so_far(self):
        return "\n\n".join([node.story_section for node in self.path_from_root()])

    def path_from_root(self):
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


# TODO: create abstract base class for simulators
# TODO: when an action is taken, generate an outline for n possible outcomes.
# Reason about how likely each outcome is and assign a probability to each. Then
# generate the next node by sampling from the distribution of outcomes.
class TextAdventureSimulator:
    def __init__(self, llm, story_request: str = None):
        self.llm = llm
        self.story_guide = self._generate_story_guide(story_request)
        self.root_node = StoryNode(self.story_guide, self.llm)
        self.current_node = self.root_node

    def _generate_story_guide(self, story_request: str) -> str:
        story_concept_generator = LLMFunction(story_concept_prompt, self.llm)
        response = story_concept_generator.generate(story_request=story_request)
        return response["guide"]

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
            action = self.current_node.actions[action_index - 1]
            action_str = f"{action.name}: {action.description}"
            self.current_node.children[action_index] = StoryNode(
                self.story_guide,
                self.llm,
                previous_action=action_str,
                parent=self.current_node
            )

        self.current_node = self.current_node.children[action_index]
        return self.state

    def is_story_ended(self) -> bool:
        return len(self.current_node.actions) == 0