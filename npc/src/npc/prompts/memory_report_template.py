from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an expert cognitive scientist specializing in memory consolidation and retrieval processes.",
    "Your task is to create a psychologically plausible memory report based on the given working memory and retrieved associations.",
]

USER_PROMPT = [
    "Memory consolidation links new observations with specific retrieved memories and current working memory."
    " Base all analysis ONLY on the provided working memory, retrieved memories, and current observation - do not assume or invent additional experiences or knowledge.",
    "",
    "Given the current working memory in <working_memory> tags and retrieved memories in <retrieved_memories> tags, analyze and synthesize using this format.",
    "IMPORTANT: Only reference information explicitly present in the provided memory contents and observation.",
    "Wrap your response in <memory_report> tags and ensure the content follows this JSON structure:",
    "{",
    "  \"patterns\": {",
    "    \"similarities\": \"Key similarities to past experience (1-2 lines)\",",
    "    \"differences\": \"Important differences (1-2 lines)\",",
    "    \"insights\": \"Lessons from past that applies (1-2 lines)\"",
    "  },",
    "  \"integration\": {",
    "    \"reinforced\": \"If applicable, beliefs/patterns this confirms (1-2 lines)\",",
    "    \"updated\": \"If applicable, ways understanding has shifted (1-2 lines)\"",
    "  },",
    "  \"emotional\": {",
    "    \"response\": \"Core emotional reaction (1 line)\",",
    "    \"relevance\": \"Why this matters to goals (1-2 lines)\"",
    "  },",
    "  \"behavior\": {",
    "    \"apply\": \"If applicable, proven strategies to use (1-2 lines)\",",
    "    \"adapt\": \"If applicable, any needed behavioral adjustments (1-2 lines)\"",
    "  }",
    "}",
    "",
    "Example format:",
    "<memory_report>",
    "{",
    "  // Your JSON content here",
    "}",
    "</memory_report>",
    "",
    "Write in first-person using natural language that reflects the agent's perspective.",
    "",
    "Here is the current working memory and retrieved memories for reference:",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<observation>",
    "{observation}",
    "</observation>",
    "",
    "<retrieved_memories>",
    "{retrieved_memories}",
    "</retrieved_memories>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "observation", "retrieved_memories"],
    output_tag_patterns=[TagPattern("memory_report")],
)
