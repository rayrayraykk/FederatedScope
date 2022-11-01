from federatedscope.register import register_optimizer


def call_my_optimizer(model, type, lr, **kwargs):
    """
    Call my optimizer if type == ' a ' else call my optimizer.

    Args:
        model: write your description
        type: write your description
        lr: write your description
    """
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        optimizer = None

    if type == 'myoptimizer':
        if optim is not None:
            optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
        return optimizer


register_optimizer('myoptimizer', call_my_optimizer)
