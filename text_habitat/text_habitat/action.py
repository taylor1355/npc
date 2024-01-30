import pprint

from .utils import PPRINT_WIDTH, execute_state_updating_code

class Action:
    def __init__(self, start_time, end_time, state_updating_code, acting_agent, metadata):
        self.start_time = start_time
        self.end_time = end_time
        self.state_updating_code = state_updating_code
        self.acting_agent = acting_agent
        self.metadata = metadata
    
    def execute(self, simulator):
        return execute_state_updating_code(self.state_updating_code, simulator) 
    
    def is_done(self, timestep):
        return timestep >= self.end_time

    def memory_str(self):
        return "\n".join([
            f"Time spent: {self.end_time - self.start_time} minutes",
            f"Description: {self.metadata.get('description', '')}", # TODO: bad practice to hardcode this key
        ])

    def __str__(self):
        return "\n".join([
            f"Action:",
            f"Time period: t={self.start_time} to t={self.end_time} minutes",
            f"State-updating code: {self.state_updating_code}",
            f"Acting agent: {self.acting_agent.name}",
            f"Metadata: {pprint.pformat(self.metadata, sort_dicts=False, width=PPRINT_WIDTH)}",
        ])