from .prompt_common import Prompt, TagPattern, create_prompt_template

SYSTEM_PROMPT = [
    "You are an expert cognitive scientist specializing in human associational memory and information retrieval processes.",
    "Your task is to formulate psychologically plausible search queries over a person's long-term memory based on their current cognitive state.",
]

USER_PROMPT = [
    "Given a written summary of the contents of a person's current working memory in <working_memory> tags and their current observation in <observation>,"
    " formulate search queries over the person's long-term memory to find relevant memories."
    " The long-term memory is implemented as a vector database of semantic text-embeddings."
    " Output the queries you decide on in <query_1></query_1>, <query_2></query_2>, ..., <query_n></query_n> tags.",
    "",
    "While formulating the queries:",
    "1. Generate a diverse set covering the full range of associations the person would likely make in about 10 seconds.",
    "2. Provide between 3 and 10 queries.",
    "3. Consider the emotional salience of information in the working memory and observation.",
    "4. Reflect the person's current goals and motivations in the query formulation.",
    "5. Account for common cognitive biases such as the recency effect, availability heuristic, and confirmation bias.",
    "6. Include both specific episodic memory queries and broader semantic knowledge queries.",
    "7. Consider the temporal context and how it might influence memory retrieval.",
    "8. Incorporate metacognitive aspects, such as the person's awareness of their own knowledge gaps.",
    "",
    "Ensure that your queries are psychologically plausible and reflect the complex, associative nature of human memory retrieval.",
    "",
    "It is essential to craft the queries in a way that will work well with retrieval-oriented contrastively trained text encodings."
    " - Instead of making your query something like 'Memory related to X', try to make it a short phrase that would be close to X in the embedding space.",
    "",
    "Here is the current working memory and observation for reference:",
    "<working_memory>",
    "{working_memory}",
    "</working_memory>",
    "",
    "<observation>",
    "{observation}",
    "</observation>",
]

prompt = Prompt(
    create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["working_memory", "observation"],
    output_tag_patterns=[TagPattern(r"query_\d+", name="queries", templated=True)],
)
