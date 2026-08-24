import copy
import re

PPRINT_WIDTH = 120
VERBOSE = True

def execute_state_updating_code(state_updating_code, room_state):
    room_state_dict = copy.deepcopy(room_state.state_dict)
    
    agent_moves = {}
    def move_agent(agent_id, new_room_id, location_in_new_room):
        room_state_dict["agents"][agent_id]["location"] = location_in_new_room
        agent_moves[agent_id] = new_room_id

    for line in state_updating_code.split("\n"):
        try:
            exec(line)
        except Exception as e:
            if VERBOSE:
                print(f"Warning: state_updating_code (line '{line}') failed to execute.")
                print(e)
        
    return room_state_dict, agent_moves


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