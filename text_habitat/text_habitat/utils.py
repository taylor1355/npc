import copy
import pprint
import re

PPRINT_WIDTH = 120
VERBOSE = True

# Idea: preprocess the code when creating the action using another LLM to write unit tests and fix any errors
def execute_state_updating_code(state_updating_code, room, agent):
    room_state_dict = copy.deepcopy(room.to_dict())
    agent_state_dict = copy.deepcopy(agent.state_dict)

    for line in state_updating_code.split("\n"):
        try:
            exec(line)
        except Exception as e:
            if VERBOSE:
                print(f"Warning: state_updating_code (line '{line}') failed to execute.")
                print(e)
        
    return room_state_dict, agent_state_dict


def extract_tags(text, defaults=None):
    tag_regex = r'<([^>/]+)>(.*?)</\1>'
    tags = re.findall(tag_regex, text, re.DOTALL)
    tags = {tag[0]: tag[1].strip() for tag in tags}

    if defaults is not None:
        for tag_name, default_value in defaults.items():
            if not tag_name in tags:
                tags[tag_name] = default_value
                if VERBOSE:
                    print(f"Warning: tag '{tag_name}' not found in LLM output. Using default value '{default_value}'.")

    return tags


def state_dict_str(state_dict):
    return pprint.pformat(state_dict, sort_dicts=False, width=PPRINT_WIDTH)