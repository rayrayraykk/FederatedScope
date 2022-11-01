from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.utils import move_to


class NLPTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        """
        Transform the data in ctx. data_batch into a loss batch and store the result in ctx

        Args:
            self: write your description
            ctx: write your description
        """
        x, label = [move_to(_, ctx.device) for _ in ctx.data_batch]
        if isinstance(x, dict):
            pred = ctx.model(**x)[0]
        else:
            pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)


def call_nlp_trainer(trainer_type):
    """
    Call the appropriate trainer function based on trainer_type.

    Args:
        trainer_type: write your description
    """
    if trainer_type == 'nlptrainer':
        trainer_builder = NLPTrainer
        return trainer_builder


register_trainer('nlptrainer', call_nlp_trainer)
