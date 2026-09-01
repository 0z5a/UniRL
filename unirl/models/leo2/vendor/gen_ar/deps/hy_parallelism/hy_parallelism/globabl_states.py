from dataclasses import dataclass


@dataclass
class GlobalStates:
    use_ptm_moe: bool = False
    use_titan_moe: bool = False

    def __post_init__(self):
        assert not (self.use_ptm_moe and self.use_titan_moe), "use_ptm_moe and use_titan_moe cannot be true at the same time"

__global_states_dict = {'default': GlobalStates()}

def get_global_states(key='default'):
    return __global_states_dict[key]