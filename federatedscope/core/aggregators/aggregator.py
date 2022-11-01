from abc import ABC, abstractmethod


class Aggregator(ABC):
    def __init__(self):
        """
        Initialize the class with the default values.

        Args:
            self: write your description
        """
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            self: write your description
            agg_info: write your description
        """
        pass


class NoCommunicationAggregator(Aggregator):
    """Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        """
        Aggregate the results.

        Args:
            self: write your description
            agg_info: write your description
        """
        # do nothing
        return {}
