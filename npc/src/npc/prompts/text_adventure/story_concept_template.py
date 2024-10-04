from npc.prompts.common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an experienced author and game designer specializing in interactive fiction and choose-your-own-adventure stories.",
    "Your task is to generate engaging story concepts and provide guidance for creating compelling interactive narratives.",
]

USER_PROMPT = [
    "Generate a concept for an interactive choose-your-own-adventure story. Consider the following aspects:",
    "1. Genre and setting",
    "2. Main character(s) and their initial situation",
    "3. Central conflict or goal",
    "4. Potential branching paths and decision points",
    "5. Possible endings",
    "",
    "Workshop a couple of ideas, exploring their potential for interactivity and player engagement.",
    "Then, create a comprehensive guide for writing the story, including style, tone, characters, plot, and worldbuilding.",
    "This guide should pertain specifically to the story concept you've developed. Think of it as a cheat sheet for writing the story.",
    "",
    "Provide your response in the following format:",
    "<workshopping>",
    "[Your workshopping process here]",
    "</workshopping>",
    "",
    "<guide>",
    "[Your comprehensive story guide here]",
    "</guide>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=[],
    output_tag_patterns=[
        TagPattern("workshopping"),
        TagPattern("guide"),
    ],
)