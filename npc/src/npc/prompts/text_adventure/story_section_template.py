from npc.prompts.common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an experienced author and game designer specializing in interactive fiction and choose-your-own-adventure narratives.",
    "Your task is to generate engaging story sections and meaningful choices for the reader based on the provided story guide and previous sections.",
]

# TODO: allow updating the guide with new information / planning for future sections
USER_PROMPT = [
    "Using the story guide, previously written sections, and the most recent action taken, create the next part of the interactive story.",
    "Ensure that your writing is consistent with the established style, tone, and world, while reflecting the consequences of the chosen action.",
    "Provide meaningful choices that advance the plot and consider the impact of previous decisions.",
    "",
    "Here is the story guide, previously written sections, and the most recent action for reference:",
    "<guide>",
    "{guide}",
    "</guide>",
    "",
    "<previous_sections>",
    "{previous_sections}",
    "</previous_sections>",
    "",
    "<previous_action>",
    "{previous_action}",
    "</previous_action>",
    "",
    "Provide your response in one of the following formats:",
    "",
    "## If the story is continuing:",
    "",
    "<next_section>",
    "[Your next story section here, incorporating the consequences of the previous action]",
    "</next_section>",
    "",
    "Then, provide between 2 and 5 possible actions for the reader to choose from. For each action:",
    "1. Make the action name concise and engaging.",
    "2. Provide a brief description that hints at potential consequences or developments.",
    "3. Ensure actions are distinct and offer meaningful choices.",
    "4. Consider the current narrative context, character motivations, and the impact of the previous action.",
    "",
    "Output the actions in <action_1>, <action_2>, ..., <action_n> tags, with nested <name> and <description> tags:",
    "<action_1>",
    "<name>[Action Name]</name>",
    "<description>[Action Description]</description>",
    "</action_1>",
    "",
    "## If the story has concluded:",
    "",
    "<epilogue>",
    "[Your story epilogue here, taking into account the final action taken]",
    "</epilogue>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["guide", "previous_sections", "previous_action"],
    output_tag_patterns=[
        TagPattern("next_section"),
        TagPattern(r"action_\d+", name="actions", multimatch=True),
        TagPattern("epilogue"),
    ],
)