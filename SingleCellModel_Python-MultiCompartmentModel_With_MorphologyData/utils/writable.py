"""
from lea
"""
from abc import ABC, abstractmethod


class Writable(ABC):

    @abstractmethod
    def get_name(self):
        """
        Must be overridden for using in the pipeline during writing the result.
        """
        pass