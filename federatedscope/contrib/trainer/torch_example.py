from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer

# An example for converting torch training process to FS training process

# Refer to `federatedscope.core.trainers.BaseTrainer` for interface.

# Try with FEMNIST:
#  python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml \
#  trainer.type mytorchtrainer federate.sample_client_rate 0.01 \
#  federate.total_round_num 5 eval.best_res_update_round_wise_key test_loss


class MyTorchTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        """
        Initialize the SGD Optimizer and Criterion

        Args:
            self: write your description
            model: write your description
            data: write your description
            device: write your description
        """
        import torch
        # NN modules
        self.model = model
        # FS `ClientData` or your own data
        self.data = data
        # Device name
        self.device = device
        # kwargs
        self.kwargs = kwargs
        # Criterion & Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=1e-4)

    def train(self):
        """
        Train the model on the device.

        Args:
            self: write your description
        """
        # _hook_on_fit_start_init
        self.model.to(self.device)
        self.model.train()

        total_loss = num_samples = 0
        # _hook_on_batch_start_init
        for x, y in self.data['train']:
            # _hook_on_batch_forward
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            # _hook_on_batch_backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # _hook_on_batch_end
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]

        # _hook_on_fit_end
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def evaluate(self, target_data_split_name='test'):
        """
        Evaluate the model on the data.

        Args:
            self: write your description
            target_data_split_name: write your description
        """
        import torch
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            total_loss = num_samples = 0
            # _hook_on_batch_start_init
            for x, y in self.data[target_data_split_name]:
                # _hook_on_batch_forward
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)

                # _hook_on_batch_end
                total_loss += loss.item() * y.shape[0]
                num_samples += y.shape[0]

            # _hook_on_fit_end
            return {
                f'{target_data_split_name}_loss': total_loss,
                f'{target_data_split_name}_total': num_samples,
                f'{target_data_split_name}_avg_loss': total_loss /
                float(num_samples)
            }

    def update(self, model_parameters, strict=False):
        """
        Updates the model parameters and returns the model para.

        Args:
            self: write your description
            model_parameters: write your description
            strict: write your description
        """
        self.model.load_state_dict(model_parameters, strict)
        return self.get_model_para()

    def get_model_para(self):
        """
        Returns the model s parameters as a dictionary.

        Args:
            self: write your description
        """
        return self.model.cpu().state_dict()


def call_my_torch_trainer(trainer_type):
    """
    Call MyTorchTrainer if trainer_type is a string.

    Args:
        trainer_type: write your description
    """
    if trainer_type == 'mytorchtrainer':
        return MyTorchTrainer


register_trainer('mytorchtrainer', call_my_torch_trainer)
