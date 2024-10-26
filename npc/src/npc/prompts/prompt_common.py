import re
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from typing import List, Union
from dataclasses import make_dataclass


def convert_buffer_to_str(buffer: Union[str, List[str]]) -> str:
    if isinstance(buffer, str):
        return buffer
    return "\n".join(buffer)


def create_prompt_template(system_prompt: Union[str, List[str]], user_prompt: Union[str, List[str]]) -> ChatPromptTemplate:
    return ChatPromptTemplate(message_templates=[
        ChatMessage(content=convert_buffer_to_str(system_prompt), role=MessageRole.SYSTEM),
        ChatMessage(content=convert_buffer_to_str(user_prompt), role=MessageRole.USER),
    ])


# TODO: Use this built in parsing functionality to allow for more complex tag patterns and nested tags: https://docs.llamaindex.ai/en/stable/examples/output_parsing/llm_program/
class TagPattern:
    def __init__(self, pattern: str, name: str = None, multimatch: bool = False):
        self.pattern = pattern

        self.name = name
        if self.name is None:
            self.name = pattern

        self.multimatch = multimatch

    def extract_from(self, text: str) -> Union[str, list[str]]:
        if self.multimatch:
            matching_tags = re.findall(f"<({self.pattern})>", text, re.DOTALL)
            return [self._extract_tag_content(text, tag) for tag in matching_tags]
        return self._extract_tag_content(text, self.pattern)

    @staticmethod
    def _extract_tag_content(text: str, tag: str) -> str:
        content_pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.search(content_pattern, text, re.DOTALL)
        if matches:
            return matches.group(1).strip()
        return None


class Prompt:
    RESPONSE_KEY = "response"

    def __init__(self,
        template: ChatPromptTemplate,
        input_tags: List[str],
        output_tag_patterns: List[TagPattern]
    ):
        self.template = template
        self.input_tags = input_tags
        self.output_tag_patterns = output_tag_patterns

        if any(pattern.name == self.RESPONSE_KEY for pattern in self.output_tag_patterns):
            raise ValueError(f"Output tag patterns cannot have the reserved name '{self.RESPONSE_KEY}'.")

    def format(self, **input_tag_contents) -> str:
        missing_tags = set(self.input_tags) - set(input_tag_contents.keys())
        if missing_tags:
            raise ValueError(f"Missing required input tags: {missing_tags}")
        return self.template.format_messages(**input_tag_contents)

    def parse_output(self, output: str) -> dict[str, Union[str, List[str]]]:
        return {
            self.RESPONSE_KEY: output,
            **self._extract_output_tags(output),
        }

    def _extract_output_tags(self, output: str) -> dict[str, Union[str, List[str]]]:
        return {tag_pattern.name: tag_pattern.extract_from(output) for tag_pattern in self.output_tag_patterns}
