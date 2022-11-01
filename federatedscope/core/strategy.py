import sys


class Strategy(object):
    def __init__(self, stg_type=None, threshold=0):
        """
        Initialize the thresholding object.

        Args:
            self: write your description
            stg_type: write your description
            threshold: write your description
        """
        self._stg_type = stg_type
        self._threshold = threshold

    @property
    def stg_type(self):
        """
        Returns the type of the STG as a string.

        Args:
            self: write your description
        """
        return self._stg_type

    @stg_type.setter
    def stg_type(self, value):
        """
        Set the stg_type of the dataset.

        Args:
            self: write your description
            value: write your description
        """
        self._stg_type = value

    @property
    def threshold(self):
        """
        Returns the threshold for the current job.

        Args:
            self: write your description
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """
        Set the threshold value.

        Args:
            self: write your description
            value: write your description
        """
        self._threshold = value
