from npc.prompts.common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are a cognitive neuroscientist specializing in working memory dynamics and attentional processes.",
    "Your task is to update the agent's working memory in a psychologically plausible manner based on the given memory report and current working memory contents.",
]

USER_PROMPT = [
    "Recent research has shown that working memory is not just a passive storage system but an active process involving the prefrontal cortex and basal ganglia."
    " It has limited capacity (typically 4-7 items) and is subject to decay and interference.",
    "",
    "Given the memory report in <memory_report> tags and the current working memory in <working_memory> tags,"
    " update the working memory and provide the result in <updated_working_memory> tags. In this process:",
    "1. Maintain a limited capacity of 4-7 main items or chunks of information.",
    "2. Prioritize new, emotionally salient, and goal-relevant information from the memory report.",
    "3. Integrate new information with existing knowledge in working memory.",
    "4. Allow for some decay of older or less relevant information.",
    "5. Account for recency and primacy effects in information retention.",
    "6. Reflect the influence of attentional focus on working memory contents.",
    "7. Include meta-cognitive elements such as current goals and task-relevant strategies.",
    "8. Consider how emotional state might influence working memory updating and capacity.",
    "9. Incorporate the agent's level of arousal or stress, which can affect working memory function.",
    "10. Allow for some variability in working memory capacity and updating efficiency to reflect individual differences.",
    "11. Consider how the agent's current cognitive load might affect the updating process.",
    "12. Reflect the associative nature of working memory by highlighting connections between items.",
    "",
    "Ensure that the report is concise, well-organized, psychologically plausible, and supports effective cognitive processing for the agent.",
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
