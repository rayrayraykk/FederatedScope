import collections
import json


def format_log_hooks(hooks_set):
    def format_dict(target_dict):
        print_dict = collections.defaultdict(list)
        for k, v in target_dict.items():
            for element in v:
                print_dict[k].append(element.__name__)
        return print_dict

    if isinstance(hooks_set, list):
        print_obj = [format_dict(_) for _ in hooks_set]
    elif isinstance(hooks_set, dict):
        print_obj = format_dict(hooks_set)
    return json.dumps(print_obj, indent=2).replace('\n', '\n\t')


def filter_by_specified_keywords(param_name, filter_keywords):
    """
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    """
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve
