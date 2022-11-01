from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer


# Build your trainer here.
class MyTrainer(GeneralTorchTrainer):
    pass


def call_my_trainer(trainer_type):
    """
    Call MyTrainer if trainer_type is a string.

    Args:
        trainer_type: write your description
    """
    if trainer_type == 'mytrainer':
        trainer_builder = MyTrainer
        return trainer_builder


register_trainer('mytrainer', call_my_trainer)
