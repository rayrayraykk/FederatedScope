class ReIterator:
    def __init__(self, loader):
        """
        Initializes the loader.

        Args:
            self: write your description
            loader: write your description
        """
        self.loader = loader
        self.iterator = iter(loader)
        self.reset_flag = False

    def __iter__(self):
        """
        Return an iterator over the elements in the collection.

        Args:
            self: write your description
        """
        return self

    def __next__(self):
        """
        Returns the next item.

        Args:
            self: write your description
        """
        try:
            item = next(self.iterator)
        except StopIteration:
            self.reset()
            item = next(self.iterator)
        return item

    def reset(self):
        """
        Reset the iterator to the original iterator.

        Args:
            self: write your description
        """
        self.iterator = iter(self.loader)
