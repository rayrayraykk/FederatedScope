from federatedscope.core.auxiliaries.model_builder import get_model


def merge_cfg(cfg, configuration, fidelity):
    init_cfg = cfg.clone()
    # Configuration related
    if 'lr' in configuration:
        init_cfg.optimizer.lr = configuration['lr']
    if 'wd' in configuration:
        init_cfg.optimizer.weight_decay = configuration['wd']
    if 'dropout' in configuration:
        init_cfg.model.dropout = configuration['dropout']
    if 'batch' in configuration:
        init_cfg.data.batch_size = configuration['batch']
    if 'layer' in configuration:
        init_cfg.model.layer = configuration['layer']
    if 'hidden' in configuration:
        init_cfg.model.hidden = configuration['hidden']
    if 'step' in configuration:
        init_cfg.federate.local_update_steps = int(configuration['step'])
    # FedOPT related
    if 'momentumsserver' in configuration:
        init_cfg.fedopt.momentum_server = configuration['momentumsserver']
    if 'lrserver' in configuration:
        init_cfg.fedopt.lr_server = configuration['lrserver']
    # Fidelity related
    if 'sample_client' in fidelity:
        init_cfg.federate.sample_client_rate = fidelity['sample_client']
    if 'round' in fidelity:
        init_cfg.federate.total_round_num = fidelity['round']
    return init_cfg


def get_cost_model(mode='estimated'):
    r"""
    This function returns a function of cost model.

    :param key: name of cost model.
    :return: the function of cost model
    """
    cost_dict = {
        'raw': raw_cost,
        'estimated': estimated_cost,
        'cross_deivce': cd_cost
    }
    return cost_dict[mode]


def communication_csilo_cost(cfg, model_size, fhb_cfg):
    t_up = model_size / fhb_cfg.cost.bandwidth.client_up
    t_down = max(
        cfg.federate.client_num * cfg.federate.sample_client_rate *
        model_size / fhb_cfg.cost.bandwidth.server_up,
        model_size / fhb_cfg.cost.bandwidth.client_down)
    return t_up + t_down


def cost_constant(configuration):
    # TODO: implement this
    return 1


def communication_cdevice_cost(cfg, model_size, fhb_cfg):
    t_up = model_size / fhb_cfg.cost.bandwidth.client_up
    t_down = model_size / fhb_cfg.cost.bandwidth.client_down
    return t_up + t_down


def computation_cost(cfg, fhb_cfg, configuration):
    """
    Assume the time is exponential distribution with c,
    return the expected maximum of M iid random variables plus server time.
    """
    t_client = sum([
        1.0 / i for i in range(
            1,
            int(cfg.federate.client_num * cfg.federate.sample_client_rate) + 1)
    ]) * cost_constant(configuration)
    return t_client + fhb_cfg.cost.time_server


def raw_cost(**kwargs):
    return None


def get_info(cfg, configuration, fidelity, data):
    cfg = merge_cfg(cfg, configuration, fidelity)
    model = get_model(cfg.model, list(data.values())[0])
    model_size = sum([param.nelement() for param in model.parameters()])
    return cfg, model_size


def estimated_cost(cfg, configuration, fidelity, data, **kwargs):
    """
    Works on raw, tabular and surrogate mode.
    """
    cfg, num_param = get_info(cfg, configuration, fidelity, data)
    t_comm = communication_csilo_cost(cfg, num_param, kwargs['fhb_cfg'])
    t_comp = computation_cost(cfg, kwargs['fhb_cfg'], configuration)
    t_round = t_comm + t_comp
    return t_round * cfg.federate.total_round_num


def cd_cost(cfg, configuration, fidelity, data, **kwargs):
    cfg, num_param = get_info(cfg, configuration, fidelity, data)
    t_comm = communication_cdevice_cost(cfg, num_param, kwargs['fhb_cfg'])
    t_comp = computation_cost(cfg, kwargs['fhb_cfg'])
    t_round = t_comm + t_comp
    return t_round * cfg.federate.total_round_num
