from federatedscope.core.monitors.monitor import Monitor


class Worker(object):
    """
    The base worker class, the parent of ``BaseClient`` and ``BaseServer``

    Args:
        ID: ID of worker
        state: the training round index
        config: the configuration of FL course
        model: the model maintained locally

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics
    """
    def __init__(self, ID=-1, state=0, config=None, model=None, strategy=None):
        """
        Initialize a new monitored object.

        Args:
            self: write your description
            ID: write your description
            state: write your description
            config: write your description
            model: write your description
            strategy: write your description
        """
        self._ID = ID
        self._state = state
        self._model = model
        self._cfg = config
        self._strategy = strategy
        self._mode = self._cfg.federate.mode.lower()
        self._monitor = Monitor(config, monitored_object=self)

    @property
    def ID(self):
        """
        ID of the object.

        Args:
            self: write your description
        """
        return self._ID

    @ID.setter
    def ID(self, value):
        """
        Set ID of the object.

        Args:
            self: write your description
            value: write your description
        """
        self._ID = value

    @property
    def state(self):
        """
        The state of the camera.

        Args:
            self: write your description
        """
        return self._state

    @state.setter
    def state(self, value):
        """
        Set the state of the motor.

        Args:
            self: write your description
            value: write your description
        """
        self._state = value

    @property
    def model(self):
        """
        The model object for this view.

        Args:
            self: write your description
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Set the model of the object.

        Args:
            self: write your description
            value: write your description
        """
        self._model = value

    @property
    def strategy(self):
        """
        Strategy used for generating output.

        Args:
            self: write your description
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        """
        Set strategy.

        Args:
            self: write your description
            value: write your description
        """
        self._strategy = value

    @property
    def mode(self):
        """
        Returns the mode of the data source.

        Args:
            self: write your description
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """
        Set the mode of the device.

        Args:
            self: write your description
            value: write your description
        """
        self._mode = value
