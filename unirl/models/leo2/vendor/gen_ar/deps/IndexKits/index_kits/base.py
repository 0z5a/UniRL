# Define the base class of various index classes.
#
from abc import ABC, abstractmethod

class IndexBase(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def shuffle(self, seed=None, fast=False, use_cache=False, save_cache=False, info=None):
        pass

    @abstractmethod
    def get_arrow_file(self, ind, **kwargs):
        pass

    @abstractmethod
    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, **kwargs):
        pass

    @abstractmethod
    def get_attribute(self, ind, column, **kwargs):
        pass

    @abstractmethod
    def get_columns(self, ind, **kwargs):
        pass

    @abstractmethod
    def random_dindex(self, ref_ind, seed=None, intra_bucket=True):
        pass
