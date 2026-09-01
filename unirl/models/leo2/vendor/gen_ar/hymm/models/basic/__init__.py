from dataclasses import dataclass
from enum import IntFlag


@dataclass
class BaseConfig:
    name: str = ""

    def get(self, key, default=None):
        return getattr(self, key, default)


class TokenMode(IntFlag):
    # We define TokenMode in a IntFlag way to support complex combinations of token modes.
    # Make sure the values are power of 2.
    DUMMY = 0b_0000_0000
    TEXT = 0b_0000_0001
    GEN_IMAGE = 0b_0000_0100
    UND_IMAGE = 0b_0000_1000
    SRC_IMAGE = 0b_0001_0000
