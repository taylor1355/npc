from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an experienced author and game designer specializing in interactive fiction and choose-your-own-adventure stories.",
    "Your task is to generate engaging story concepts and provide guidance for creating compelling interactive narratives.",
]

USER_PROMPT = [
    "Generate a concept for an interactive choose-your-own-adventure story. Consider the following aspects:",
    "1. Genre and setting.",
    "2. Detailed worldbuilding elements. This should be quite specific and sketch out the world in which the story takes place.",
    "3. Stylistic choices and tone. Be detailed here, lay out a clear vision.",
    "4. Central conflict or goal(s). There may be multiple conflicts or goals, the interplay of which may be important.",
    "5. Main character(s) and their initial situation.",
    "6. Pace and structure of the story. How will you keep the reader engaged and allow the conflict to be resolved in a reasonable timeframe?"
    " This should be a one-off story, with a clear beginning, middle, and end. However, note that there will be many branching paths so there"
    " won't be a single linear narrative. Instead, you are describing a family of stories that all start the same but can end in many different ways."
    " Another important point is that the story may end early depending on the reader's choices. It is important to not drag out the story."
    "7. Some example branching paths and decision points that fit well with the story concept. Make it clear that these are just examples and that"
    " the reader's choices will determine the actual path they take.",
    "8. Some example endings that would fit well with the story concept. Make it clear that these are just examples and that the reader's choices"
    " will determine the actual ending they get.",
    "",
    "Workshop a couple of ideas, exploring their potential for interactivity and player engagement. All ideas should adhere completely to"
    " the user's story request in <story_request> tags. Treat the story request as a set of specifications that your story concept must meet."
    " Then, create a comprehensive guide for writing the story you come up with, including style, tone, characters, plot, and worldbuilding.",
    "- Make sure the story concept ideas are unique, engaging, and well-suited for an interactive narrative.",
    "- This guide should pertain specifically to the story concept you've developed. Think of it as a cheat sheet for writing the story.",
    "- Focus on packing in as much information as possible to flesh out the foundation of the story.",
    "- Make sure that the guide contains all specifications from the story request and provides clear direction for writing the story.",
    "",
    "Here is the story request, which you must adhere to in your story concept and guide:",
    "<story_request>",
    "{story_request}",
    "</story_request>",
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
    input_tags=["story_request"],
    output_tag_patterns=[
        TagPattern("workshopping"),
        TagPattern("guide"),
    ],
)