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
    "Analyze the emails in <email_analysis> tags.",
    "For each email summary:",
    "- Identify any action items it contains",
    "- If this email contains interesting information, what would the user have learned from carefully reading the email?",
    "- Critically think about whether to include any links mentioned (only if the user is quite likely to want to click them)",
    "- What are some properties this email shares in common with other emails? This can he helpful for grouping emails together.",
    "- Strategize how this information should be integrated into the briefing.",
    "",
    "Now, plan how you will structure the briefing to communicate the information to the user in a low-cognitive-load, high-impact way.", 
    "",
    "# 2. Create an outline:",
    "Craft an outline inside <outline> tags. First, give the name and structure of each section and sub-section.",
    " Then, provide a description of each section and sub-section, starting from the last sub-section and working your way up to the top-level sections.",
    "- The first section should be the table of contents.",
    "- The second section should be a short executive summary of everything in the briefing. Low-importance details should aggregated together and heavily summarized.",
    "- Each section should begin with a section summary.",
    "- Include separate sections for distinct topics or areas of focus.",
    "- Group related topics together.",
    "",
    "# 3. Write the briefing:",
    "Following the structure from the outline, create the briefing in <briefing> tags with the following requirements:",
    "- Progressive disclosure: Start with the most important information and gradually move to less important details.",
    "- Do not just give a list of summaries. Synthesize information and provide insights, while making it clear what is purely factual and what is your analysis.",
    "- Newsletters should be described in more detail than other emails. Each topic of each newsletter should have a separate section.",
    "- Highlight any recommended action items clearly in the summary part of each section.",
    "- Maintain an efficient, concise, clear, and professional tone throughout.",
    "- Use markdown formatting for readability. You should make use of bullet points when applicable to reduce cognitive load.",
    "- Insert citations to the original email summaries using the format [<email_id>id</email_id>] when referencing them in the briefing.",
    "- Include links which would save the user time and effort, but only if they are likely to be useful.",
    "  - This will usually include unsubscribe links, but may also include links to articles, websites, or other resources.",
    "  - If you are recommending an action for the user to take, include the relevant link if applicable.",
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