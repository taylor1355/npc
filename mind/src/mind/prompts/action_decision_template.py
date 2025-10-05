from .prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an expert in cognitive psychology and decision-making processes.",
    "Your task is to determine the most likely action an agent would take based on their current working memory, personality traits, and available actions, in a psychologically plausible manner.",
]

USER_PROMPT = [
    "Given the working memory in <working_memory> tags, personality traits in <personality_traits> tags, and available actions in <available_actions> tags,"
    " analyze the situation and determine the most appropriate action. Provide your analysis in <decision_explanation> tags using this format:",
    "{",
    "  \"state\": \"One-line summary of current situation\",",
    "  \"goals\": \"One-line summary of objectives\",",
    "  \"traits\": \"One-line summary of relevant personality effects\",",
    "  \"actions\": {",
    "    \"for each action\": {",
    "      \"pros\": \"2-3 key benefits\",",
    "      \"cons\": \"2-3 key drawbacks\",",
    "      \"params\": \"If needed, parameter considerations\"",
    "    }",
    "  },",
    "  \"choice\": \"One-line explanation of selected action\"",
    "}",
    "",
    "Then provide the chosen action in <action_decision> tags as a JSON object with:",
    "- action_index: The index of the chosen action",
    "- parameters: Any parameters for the action (if applicable)",
    "",
    "Here is the current state for reference:",
    "",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<personality_traits>",
    "{personality_traits}",
    "</personality_traits>",
    "",
    "<available_actions>",
    "{available_actions}",
    "",
    "Each action has:",
    "- Index: The action's position in the list",
    "- Description: What the action does",
    "</available_actions>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "personality_traits", "available_actions"],
    output_tag_patterns=[
        TagPattern("decision_explanation"),  # Explanation of decision
        TagPattern("action_decision"),  # JSON with action index
    ]
)
