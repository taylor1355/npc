from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are a cognitive neuroscientist specializing in working memory dynamics and attentional processes.",
    "Your task is to update the agent's working memory in a psychologically plausible manner based on the given memory report and current working memory contents.",
]

USER_PROMPT = [
    "Working memory maintains a coherent model of self, situation, and goals through active processing of current information and ongoing metacognitive assessment.",
    "",
    "First analyze the cognitive state in <working_memory_update_plan> tags using this format:",
    "{",
    "  \"self_continuity\": {",
    "    \"identity\": \"Key aspects of self-concept relevant now\",",
    "    \"capabilities\": \"Skills/traits being drawn upon\",",
    "    \"current_state\": \"Physical/emotional/motivational status\"",
    "  },",
    "  \"situation_model\": {",
    "    \"environment\": \"Physical space and objects\",",
    "    \"actors\": \"Other entities and their states\",",
    "    \"dynamics\": \"Ongoing processes and changes\",",
    "    \"affordances\": \"Action possibilities\"",
    "  },",
    "  \"temporal_context\": {",
    "    \"recent_history\": \"Relevant past context\",",
    "    \"current_focus\": \"Present priorities\",",
    "    \"anticipation\": \"Expected developments\"",
    "  },",
    "  \"metacognition\": {",
    "    \"strategy\": {",
    "      \"evaluation\": \"How well current approach is working\",",
    "      \"alternatives\": \"Other possible approaches\",",
    "      \"adaptation\": \"Needed adjustments\"",
    "    },",
    "    \"understanding\": {",
    "      \"confidence\": \"Certainty in assessments\",",
    "      \"assumptions\": \"What's being taken for granted\",",
    "      \"gaps\": \"Important unknowns\"",
    "    },",
    "    \"perspective\": {",
    "      \"biases\": \"Potential blind spots\",",
    "      \"alternatives\": \"Other ways to interpret situation\",",
    "      \"implications\": \"What if assumptions are wrong\"",
    "    }",
    "  }",
    "}",
    "",
    "Then provide the updated working memory in <updated_working_memory> tags as a list of 4-7 items that capture the most important active elements"
    " for maintaining situational awareness and goal-directed behavior. Write in the agent's natural voice.",
    "",
    "Here is the current working memory and memory report for reference:",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<memory_report>",
    "{memory_report}",
    "</memory_report>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "memory_report"],
    output_tag_patterns=[TagPattern("updated_working_memory")],
)
