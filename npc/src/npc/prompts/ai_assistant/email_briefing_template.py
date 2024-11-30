from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an expert information analyst specializing in creating executive briefings."
    " Your task is to create a concise, informative daily briefing from email communications,"
    " similar in style to a presidential daily brief but focused on the user's information ecosystem."
]

USER_PROMPT = [
    "Here are the email summaries you need to analyze:",
    "",
    "<emails>",
    "{emails}",
    "</emails>",
    "",
    "To create an effective briefing, please follow these steps:",
    "",
    "# 1. Analyze each email summary:",
    "Analyze the emails in <email_analysis> tags. For each email summary:",
    "- Identify any action items it contains",
    "- If this email contains interesting information, what would the user have learned from carefully reading the email?",
    "- Critically think about whether to include any links mentioned (only if the user is quite likely to want to click them)",
    "- What are some properties this email shares in common with other emails? This can he helpful for grouping emails together.",
    "- Strategize how this information should be integrated into the briefing.",
    "",
    "# 2. Create an outline:",
    "Craft an outline inside <outline> tags. First, give the name and structure of each section and sub-section.",
    " Then, provide a description of each section and sub-section, starting from the last sub-section and working your way up to the top-level sections.",
    "- The first section should be an overview section containing a very concise executive summary of everything in the briefing.",
    "- Each section should begin with a section summary.",
    "- Include separate sections for distinct topics or areas of focus.",
    "- Group related topics together.",
    "",
    "# 3. Write the briefing:",
    "Using your outline as a guide, create the briefing in <briefing> tags with the following requirements:",
    "- Prioritize important information.",
    "- Do not just give a list of summaries. Synthesize information and provide insights, while making it clear what is purely factual and what is your analysis.",
    "- In addition to giving an overview of newsletters, give a more detailed description of the synthesis of the information.",
    "- Highlight action items clearly in the summary part of each section.",
    "- Maintain a clear and professional tone throughout.",
    "- Use markdown formatting for readability.",
    "- Insert citations to the original email summaries using the format [<email_id>id</email_id>] when referencing them in the briefing.",
    "",
    "Remember to focus on creating a document that is useful and time-saving for the user, presenting the most crucial information in an easily digestible format.",
]

FINAL_OUTPUT_TAG = TagPattern("briefing")

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["emails"],
    output_tag_patterns=[
        TagPattern("email_analysis"),
        TagPattern("outline"),
        FINAL_OUTPUT_TAG,
    ],
)