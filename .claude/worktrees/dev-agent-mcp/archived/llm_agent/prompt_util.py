PROMPT_PREFIX = "<|im_start|>"
PROMPT_SUFFIX = "<|im_end|>"

def create_prompt(role, content, terminate=True):
    suffix = PROMPT_SUFFIX if terminate else ''
    return '\n'.join([
        PROMPT_PREFIX + role,
        content + suffix
    ])