import copy
import re

PPRINT_WIDTH = 120

# Idea: preprocess the code when creating the action using another LLM to write unit tests and fix any errors
def execute_state_updating_code(state_updating_code, simulator):
    room_state_dict = copy.deepcopy(simulator.room.to_dict())
    agent_state_dict = copy.deepcopy(simulator.agent.to_dict())

    for line in state_updating_code.split("\n"):
        try:
            exec(line)
        except Exception as e:
            print(f"Warning: state_updating_code (line '{line}') failed to execute.")
            print(e)
        
    return room_state_dict, agent_state_dict

def extract_tags(text):
    tag_regex = r'<([^>/]+)>(.*?)</\1>'
    tags = re.findall(tag_regex, text, re.DOTALL)
    return {tag[0]: tag[1].strip() for tag in tags}