import abc
import inspect


class BaseTrainer(abc.ABC):
    def __init__(self, model, data, device, **kwargs):
        """
        Initialize the data object

        Args:
            self: write your description
            model: write your description
            data: write your description
            device: write your description
        """
        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def train(self):
        """
        Train the model.

        Args:
            self: write your description
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, target_data_split_name='test'):
        """
        Evaluate the split.

        Args:
            self: write your description
            target_data_split_name: write your description
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model_parameters, strict=False):
        """
        Updates the model with the given model parameters.

        Args:
            self: write your description
            model_parameters: write your description
            strict: write your description
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_para(self):
        """
        Returns the model parameter for the model.

        Args:
            self: write your description
        """
        raise NotImplementedError

    def print_trainer_meta_info(self):
        """
        Returns: String contains meta information of Trainer.
        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'
