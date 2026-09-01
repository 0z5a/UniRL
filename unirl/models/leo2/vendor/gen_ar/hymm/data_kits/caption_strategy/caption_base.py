from typing import Optional
from dataclasses import dataclass


@dataclass
class CaptionOut(object):
    caption: Optional[str] = ""
    lang: Optional[str] = ""
    key: Optional[str] = ""
    sel_col: Optional[str] = ""
    tag_keys: Optional[list[str]] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        setattr(self, key, value)
