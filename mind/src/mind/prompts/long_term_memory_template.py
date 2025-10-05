from .prompt_common import create_prompt_template, Prompt, TagPattern

SYSTEM_PROMPT = [
    "You are a cognitive neuroscientist specializing in long-term memory processes, including encoding, consolidation, and retrieval.",
    "Your task is to update the agent's long-term memory in a psychologically plausible manner based on the given working memory contents and current observations.",
]

USER_PROMPT = [
    "Recent research in cognitive psychology and neuroscience has revealed that long-term memory formation is influenced by various factors such as emotional salience, repetition, association with existing knowledge, and the consolidation process during rest or sleep.",
    "Long-term memory is organized semantically and episodically, allowing for the storage of facts, concepts, and personal experiences.",
    "",
    "Given the working memory contents in <working_memory> tags and the current observation in <observation> tags, update the agent's long-term memory entries, and provide them as <memory_1></memory_1>, ..., <memory_n></memory_n>.",
    "In this process:",
    "1. Prioritize emotionally significant, novel, and goal-relevant information for encoding into long-term memory.",
    "2. Integrate new information with existing long-term memories by forming associations.",
    "3. Use elaborative encoding strategies to enhance memory consolidation.",
    "4. Organize memories semantically, grouping related concepts together.",
    "5. Consider the effects of rehearsal and repetition on memory strength.",
    "6. Account for any potential interference or forgetting of older, less relevant memories.",
    "7. Reflect the influence of the agent's attention and focus on memory encoding.",
    "8. Include meta-cognitive elements such as awareness of learning strategies or memory techniques being employed.",
    "9. Consider how the agent's emotional state and stress levels might affect memory encoding and consolidation.",
    "10. Ensure that the updated long-term memories are stored in a way that facilitates future retrieval and use.",
    "",
    "Ensure that the memory entries are concise, well-organized, psychologically plausible, and support effective cognitive functioning for the agent.",
    "The long-term memory entries should be written in the voice of the agent, reflecting their internal thoughts and knowledge.",
    "Use the first-person perspective, and avoid using technical terms from cognitive psychology or neuroscience that the agent would not know.",
    "Instead, the memories should be written in clear, everyday language that the agent would use.",
    "",
    "Here are the current working memory contents and the observation for reference:",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<observation>",
    "{observation}",
    "</observation>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "observation"],
    output_tag_patterns=[TagPattern(r"memory_\d+", name="memories", templated=True)],
)
