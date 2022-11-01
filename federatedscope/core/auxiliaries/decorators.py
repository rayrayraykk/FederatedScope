def use_diff(func):
    """
    Decorate a function to use the diff metric when federate is used.

    Args:
        func: write your description
    """
    def wrapper(self, *args, **kwargs):
        """
        Wrap the evaluation function to add diff information if needed.

        Args:
            self: write your description
        """
        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            before_metric = self.evaluate(target_data_split_name='val')

        num_samples_train, model_para, result_metric = func(
            self, *args, **kwargs)

        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            after_metric = self.evaluate(target_data_split_name='val')
            result_metric['val_total'] = before_metric['val_total']
            result_metric['val_avg_loss_before'] = before_metric[
                'val_avg_loss']
            result_metric['val_avg_loss_after'] = after_metric['val_avg_loss']

        return num_samples_train, model_para, result_metric

    return wrapper
