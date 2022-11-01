from federatedscope.register import register_metric
import numpy as np


def compute_poison_metric(ctx):
    """
    Computes the poison metric for the current context.

    Args:
        ctx: write your description
    """

    poison_true = ctx['poison_' + ctx.cur_split + '_y_true']
    poison_prob = ctx['poison_' + ctx.cur_split + '_y_prob']
    poison_pred = np.argmax(poison_prob, axis=1)

    correct = poison_true == poison_pred

    return float(np.sum(correct)) / len(correct)


def load_poison_metrics(ctx, y_true, y_pred, y_prob, **kwargs):
    """
    Computes the poisson metric for the current split.

    Args:
        ctx: write your description
        y_true: write your description
        y_pred: write your description
        y_prob: write your description
    """

    if ctx.cur_split == 'train':
        results = None
    else:
        results = compute_poison_metric(ctx)

    return results


def call_poison_metric(types):
    """
    Check if the metric is the one that should be called.

    Args:
        types: write your description
    """
    if 'poison_attack_acc' in types:
        the_larger_the_better = True
        return 'poison_attack_acc', load_poison_metrics, the_larger_the_better


register_metric('poison_attack_acc', call_poison_metric)
