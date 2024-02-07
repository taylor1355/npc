import pprint

from .utils import PPRINT_WIDTH

class State:
    def __init__(self, initial_state_dict, required_keys, immutable_keys, key_serializers=None):
        self.state_dict = initial_state_dict
        self.required_keys = required_keys
        self.immutable_keys = immutable_keys
        self.key_serializers = {} if key_serializers is None else key_serializers

        for key in self.required_keys:
            if key not in self.state_dict:
                raise ValueError(f"Key '{key}' is required in state_dict, but it is not present.")

    def update(self, new_state_dict):
        # make sure all required keys are present
        for key in self.required_keys:
            if key not in new_state_dict:
                print(f"Warning: key '{key}' is required but not found in new state_dict.")
                return

        # update the state_dict
        for key, value in new_state_dict.items():
            if key in self.state_dict and key not in self.immutable_keys and self._serialize_value_if_needed(key, value) is not None:
                self.state_dict[key] = value
            elif key not in self.state_dict:
                print(f"Warning: key '{key}' not found in state_dict.")

    def _serialize_value_if_needed(self, key, value):
        if key in self.key_serializers:
            try:
                value = self.key_serializers[key](value)
            except Exception as e:
                print(f"Warning: failed to serialize key '{key}' with value '{value}' using serializer {self.key_serializers[key]}: {e}")
                return None
        return value
            
    def __getitem__(self, key):
        return self._serialize_value_if_needed(key, self.state_dict[key])

    def __str__(self):
        return pprint.pformat(self.state_dict, sort_dicts=False, width=PPRINT_WIDTH)