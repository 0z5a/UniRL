import loguru
from collections.abc import Mapping
from functools import partial
from typing import Callable, Any, Dict, Iterator

class LazyEntry:

    def __init__(self, key: str, value_fn: Callable):
        self.key = key
        self.value_fn = value_fn
    
    def __call__(self):
        return self.value_fn()

class LazyStateDict(Mapping):
    def __init__(self, state_dict: Mapping[str, Any]):
        self._state_dict = state_dict

    def __getitem__(self, key):
        ret = self._state_dict[key]
        if isinstance(ret, LazyEntry):
            assert key == ret.key, f'{key=} != {ret.key=}'
            return ret()
        elif callable(ret):
            return ret()
        else:
            loguru.logger.warning(f'{key=} is not found in the state_dict, returning {type(ret)}')
            return ret

    def __len__(self) -> int:
        return len(self._state_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._state_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._state_dict

    def to_eager_dict(self) -> Dict[str, Any]:
        return {k: self[k] for k in self._state_dict}

class LazyMappedStateDict(Mapping):
    
    def __init__(self, state_dict: Mapping[str, Any], mapped_state_dict):
        """
        mapped_state_dict 是转换后的 sd, 包含了所有转换后的key, 如果value是callable, 则会在调用时自动绑定参数
        如果是 None, 则 fallback 到 state_dict
        """
        self._state_dict = state_dict
        self._mapped_state_dict = mapped_state_dict

    def __getitem__(self, key):
        ret = self._mapped_state_dict.get(key)
        if isinstance(ret, LazyEntry):
            assert key == ret.key, f'{key=} != {ret.key=}'
            return ret()
        elif callable(ret):
            return ret()
        elif ret is None: # None means the original
            return self._state_dict[key]
        else:
            loguru.logger.warning(f'{key=} is not found in the state_dict, returning {type(ret)}')
            return ret

    def items(self):
        for k in self._mapped_state_dict:
            yield k, self[k]
    
    def __len__(self) -> int:
        return len(self._mapped_state_dict)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._mapped_state_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._mapped_state_dict
    
    def __repr__(self) -> str:
        return f"LazyMappedStateDict({self._mapped_state_dict})"

    def to_eager_dict(self) -> Dict[str, Any]:
        return {k: self[k] for k in self._mapped_state_dict}

def parameter_mapping_fn_example(sd:dict) -> Dict[str, Any]:
    ret = {}
    for k, v in sd.items():
        def get_new_val(sd, k):
            return sd[k]
        if 'abc' in k:
            ret[f'{k}_new'] = partial(get_new_val, sd, k)
        else:
            ret[k] = None
    return ret