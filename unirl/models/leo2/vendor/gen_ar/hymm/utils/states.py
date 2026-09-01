import json
from dataclasses import dataclass, asdict


@dataclass
class DataClassMixin:

    def __repr__(self):
        max_key_length = max([len(k) for k in asdict(self).keys()])
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def legacy_to_dict(self):
        # For bc of python3.10
        from copy import deepcopy
        from collections import defaultdict

        states = deepcopy(vars(self))
        # Convert defaultdict to dict
        for key, value in states.items():
            if isinstance(value, defaultdict):
                states[key] = dict(value)
        return states

    def to_dict(self):
        try:
            return asdict(self)
        except TypeError:
            # Fallback to legacy implementation
            return self.legacy_to_dict()

    def from_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        if hasattr(self, "__post_init__"):
            self.__post_init__()

    def serialize(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)

    def deserialize(self, state):
        return self.from_dict(json.loads(state))
