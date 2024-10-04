from npc.prompts.common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an expert in cognitive psychology and decision-making processes.",
    "Your task is to determine the most likely action an agent would take based on their current working memory and available actions, in a psychologically plausible manner.",
]

USER_PROMPT = [
    "Recent research in decision neuroscience has shown that human decision-making involves complex interactions between the prefrontal cortex, basal ganglia, and limbic system."
    " This process integrates cognitive, emotional, and motivational factors, often using heuristics and being influenced by biases.",
    "",
    "Given the working memory in <working_memory> tags and a description of the available actions in <available_actions>,"
    " choose which action the agent would be most likely to take and specify how it should be executed."
    " Respond in the format specified in <available_actions> within <action_decision> tags. In making this decision:",
    "1. Consider the agent's current goals, emotional state, and motivations as reflected in working memory.",
    "2. Evaluate the potential outcomes of each action in relation to the agent's goals.",
    "3. Account for the agent's past experiences and learned patterns of behavior.",
    "4. Incorporate elements of bounded rationality, recognizing that the agent has limited cognitive resources and may use heuristics.",
    "5. Allow for the influence of cognitive biases such as confirmation bias or availability heuristic.",
    "6. Consider the role of intuition and gut feelings in quick decision-making.",
    "7. Reflect the agent's risk tolerance and time preferences.",
    "8. Include any necessary preparation or planning steps as part of the action execution.",
    "9. Consider how the agent's current physiological state (e.g., fatigue, hunger) might influence the decision.",
    "10. Incorporate metacognitive aspects, such as the agent's confidence in their decision.",
    "11. Allow for some variability in decision-making to reflect individual differences and the probabilistic nature of human choice.",
    "12. Consider how social factors or perceived social norms might influence the decision.",
    "13. Reflect the associative nature of decision-making by considering how the chosen action might relate to or trigger other potential actions.",
    "",
    "Ensure that the chosen action and its execution are psychologically plausible, consistent with the agent's current cognitive state, and likely to be effective in pursuing the agent's goals."
    "",
    "Here is the current working memory and available actions for reference:",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<available_actions>",
    "{available_actions}",
    "",
    "# Action Decision Format",
    "In order to proceed, please provide the action decision in <action_decision> tags as a valid JSON object with the following fields:",
    "{action_request_documentation}",
    "</available_actions>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "available_actions", "action_request_documentation"],
    output_tag_patterns=[TagPattern("action_decision")]
)