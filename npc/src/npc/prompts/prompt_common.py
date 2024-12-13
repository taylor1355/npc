import re
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')

def convert_buffer_to_str(buffer: str | list[str]) -> str:
    """Convert a string or list of strings into a single string.
    
    Args:
        buffer: Either a string or list of strings to convert
        
    Returns:
        A single string with list items joined by newlines if input was a list
    """
    if isinstance(buffer, str):
        return buffer
    return "\n".join(buffer)


def create_prompt_template(system_prompt: str | list[str], user_prompt: str | list[str]) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from system and user prompts.
    
    Args:
        system_prompt: System prompt text as string or list of strings
        user_prompt: User prompt text as string or list of strings
        
    Returns:
        ChatPromptTemplate configured with the provided prompts
    """
    return ChatPromptTemplate(message_templates=[
        ChatMessage(content=convert_buffer_to_str(system_prompt), role=MessageRole.SYSTEM),
        ChatMessage(content=convert_buffer_to_str(user_prompt), role=MessageRole.USER),
    ])


# TODO: Use this built in parsing functionality to allow for more complex tag patterns and nested tags: https://docs.llamaindex.ai/en/stable/examples/output_parsing/llm_program/
class TagPattern:
    """Pattern for extracting and optionally parsing content from XML-style tags.
    
    Attributes:
        pattern: Regex pattern for the tag name to match
        name: Optional name to use as the key when storing extracted values
        templated: Whether the tag can appear multiple times with different values matching the pattern
        parser: Function to parse the extracted string value into another type
    """
    
    def __init__(self, 
                 pattern: str, 
                 name: str = None, 
                 templated: bool = False,
                 parser: Optional[Callable[[str], T]] = None):
        self.pattern = pattern
        self.name = name if name else pattern
        self.templated = templated
        self.parser = parser

    def extract_from(self, text: str) -> str | list[str] | T | list[T] | None:
        """Extract and optionally parse tag content from text.
        
        Args:
            text: The text to extract tag content from
            
        Returns:
            Extracted and optionally parsed value(s), or None if no match found.
            Returns a list if templated=True, otherwise returns a single value.
        """
        if self.templated:
            matching_tags = re.findall(f"<({self.pattern})>", text, re.DOTALL)
            tag_contents = [self._extract_tag_content(text, tag) for tag in matching_tags]

            # sort tags in alphabetical order to ensure consistent output
            sorted_tuples = sorted(zip(matching_tags, tag_contents), key=lambda x: x[0])
            sorted_contents = [tag_contents for _, tag_contents in sorted_tuples]

            return [self._parse_contents(tag_contents) for tag_contents in sorted_contents]
        
        contents = self._extract_tag_content(text, self.pattern)
        return self._parse_contents(contents) if contents else None

    def _parse_contents(self, value: str) -> str | T:
        """Parse a string value using the configured parser if one exists.
        
        Args:
            value: The string value to parse
            
        Returns:
            The parsed value if a parser is configured, otherwise the original string
        """
        if self.parser:
            return self.parser(value)
        return value

    @staticmethod
    def _extract_tag_content(text: str, tag: str) -> Optional[str]:
        """Extract the content between opening and closing tags.
        
        Args:
            text: The text to extract from
            tag: The tag name to match
            
        Returns:
            The content between the tags, or None if no match found
        """
        content_pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.search(content_pattern, text, re.DOTALL)
        if matches:
            return matches.group(1).strip()
        return None


class Prompt:
    """Template for generating and parsing LLM prompts with XML-style tags.
    
    Attributes:
        template: The ChatPromptTemplate containing the prompt structure
        input_tags: List of required input tag names
        output_tag_patterns: List of TagPattern objects for parsing the response
    """
    
    RESPONSE_KEY = "response"

    def __init__(self,
        template: ChatPromptTemplate,
        input_tags: list[str],
        output_tag_patterns: list[TagPattern]
    ):
        self.template = template
        self.input_tags = input_tags
        self.output_tag_patterns = output_tag_patterns

        if any(pattern.name == self.RESPONSE_KEY for pattern in self.output_tag_patterns):
            raise ValueError(f"Output tag patterns cannot have the reserved name '{self.RESPONSE_KEY}'.")

    def format(self, **input_tag_contents) -> str:
        """Format the prompt template with provided input values.
        
        Args:
            **input_tag_contents: Keyword arguments containing values for input tags
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If any required input tags are missing
        """
        missing_tags = set(self.input_tags) - set(input_tag_contents.keys())
        if missing_tags:
            raise ValueError(f"Missing required input tags: {missing_tags}")
        return self.template.format_messages(**input_tag_contents)

    def parse_output(self, output: str) -> dict[str, Any]:
        """Parse an LLM response using the configured output tag patterns.
        
        Args:
            output: The raw LLM response text to parse
            
        Returns:
            Dictionary mapping tag names to their extracted/parsed values.
            Includes the full response text under the reserved key Prompt.RESPONSE_KEY.
        """
        return {
            self.RESPONSE_KEY: output,
            **self._extract_output_tags(output),
        }

    def _extract_output_tags(self, output: str) -> dict[str, Any]:
        """Extract and parse all configured output tags from response text.
        
        Args:
            output: The text to extract tags from
            
        Returns:
            Dictionary mapping tag names to their extracted/parsed values
        """
        return {tag_pattern.name: tag_pattern.extract_from(output) for tag_pattern in self.output_tag_patterns}
