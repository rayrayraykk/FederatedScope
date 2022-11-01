from federatedscope.register import register_criterion


def call_my_criterion(type, device):
    """
    Call the X - Entropy Criterion to get the correct value.

    Args:
        type: write your description
        device: write your description
    """
    try:
        import torch.nn as nn
    except ImportError:
        nn = None
        criterion = None

    if type == 'mycriterion':
        if nn is not None:
            criterion = nn.CrossEntropyLoss().to(device)
        return criterion


register_criterion('mycriterion', call_my_criterion)
