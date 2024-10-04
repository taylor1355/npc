from npc.prompts.common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are an expert cognitive scientist specializing in memory consolidation and retrieval processes.",
    "Your task is to create a psychologically plausible memory report based on the given working memory and retrieved associations.",
]

USER_PROMPT = [
    "Recent neuroscience research has shown that memory consolidation involves the hippocampus and neocortex working together to integrate new information with existing knowledge."
    " This process is influenced by factors such as emotional salience, relevance to current goals, and the strength of neural connections.",
    "",
    "Given the current working memory in <working_memory> tags and retrieved memories in <retrieved_memories> tags, create a memory report in <memory_report> tags. This report should:",
    "1. Synthesize and compile all interesting, useful, relevant, and important information from the memories and the observation.",
    "2. Incorporate and refine information from the working memory.",
    "3. Prioritize information based on emotional significance and relevance to current goals.",
    "4. Reflect the associative nature of human memory by highlighting connections between different pieces of information.",
    "5. Include both episodic (event-specific) and semantic (general knowledge) elements.",
    "6. Demonstrate the reconstructive nature of memory by filling in gaps with plausible inferences.",
    "7. Account for the temporal dynamics of memory retrieval, with more recent or emotionally charged memories potentially being more vivid.",
    "8. If appropriate, include metacognitive reflections on the reliability or completeness of recalled information.",
    "9. Consider how common cognitive biases (e.g., confirmation bias, hindsight bias) might influence the memory reconstruction process.",
    "10. Reflect individual differences in memory processing by allowing for some variability in the synthesis approach.",
    "",
    "Ensure that the memory report is coherent, psychologically plausible, and supports the agent's competence in decision-making and problem-solving."
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