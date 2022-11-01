from federatedscope.register import register_model


# Build you torch or tf model class here
class MyNet(object):
    pass


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    """
    Model builder function.

    Args:
        model_config: write your description
        local_data: write your description
    """

    model = MyNet()

    return model


def call_my_net(model_config, local_data):
    """
    Call my net model builder if it is available.

    Args:
        model_config: write your description
        local_data: write your description
    """
    if model_config.type == "mynet":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("mynet", call_my_net)
