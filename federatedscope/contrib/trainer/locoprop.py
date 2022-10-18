import numpy as np

from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer

# This is an example for locoprop trainer

# Try with MNIST:
#  python federatedscope/main.py --cfg
#  scripts/example_configs/mnist_locoprop.yaml


class LocoPropTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        super(LocoPropTrainer, self).__init__(model, data, device, **kwargs)
        import torch
        self.act_lr = 10.0
        self.num_local_iters = 10
        self.lr = 0.01
        self.enable_locoprop = False
        self.round = 0
        self.warmup = 25
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=1e-4)

    def train(self):
        import torch
        import torch.nn.functional as F

        # Warmup
        if self.round <= self.warmup:
            self.enable_locoprop = False
            self.change_lr(float(self.round) / float(self.warmup) * self.lr)
        else:
            self.enable_locoprop = True
            base_lr = (self.lr * (1.0 - (self.round - self.warmup) /
                                  (self.round - self.warmup)))

        # Start training
        self.model.to(self.device)
        self.model.train()

        total_loss = num_samples = 0
        for x, y in self.data['train']:
            self.optimizer.zero_grad()
            activations, post_activations, target_gds, batch_input_layers, \
                batch_target_gds = [], [], [], [], []
            x, y = x.to(self.device), y.to(self.device)
            x = torch.flatten(x, start_dim=1).requires_grad_()
            out = self.model(x, activations, post_activations)
            loss = F.nll_loss(out, y, reduction='mean')

            if self.enable_locoprop:
                act_grad = torch.autograd.grad(loss,
                                               activations,
                                               allow_unused=True)
                for i in range(len(self.model.linears)):
                    input_to_layer = x if i == 0 else post_activations[i - 1]
                    activation = activations[i]
                    target_gd = activation - self.act_lr * act_grad[i]
                    batch_target_gds.append(target_gd.detach())
                    batch_input_layers.append(input_to_layer.detach())

                decay = np.maximum(
                    1.0 - np.arange(self.num_local_iters) /
                    float(self.num_local_iters), 0.25)
                lrs = base_lr * decay

                for t in range(self.num_local_iters):
                    self.change_lr(lrs[t])
                    self.optimizer.zero_grad()
                    for i in range(len(self.model.linears)):
                        fake_activation = self.model.linears[i](
                            batch_input_layers[i])
                        local_loss = F.mse_loss(fake_activation,
                                                batch_target_gds[i],
                                                reduction='mean')
                        grads = torch.autograd.grad(
                            local_loss, self.model.linears[i].parameters())
                        for p, g in zip(self.model.linears[i].parameters(),
                                        grads):
                            p.grad = g.detach()
                    self.optimizer.step()
            else:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]

        self.round += 1

        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def evaluate(self, target_data_split_name='test'):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            total_loss = num_samples = 0
            y_pred = []
            ys = []
            for x, y in self.data[target_data_split_name]:
                x, y = x.to(self.device), y.to(self.device)
                x = torch.flatten(x, start_dim=1)
                out = self.model(x)
                loss = F.nll_loss(out, y, reduction='mean')
                pred = out.argmax(dim=-1)
                y_pred += pred.tolist()
                ys += y.detach().cpu().tolist()
                total_loss += loss.item() * y.shape[0]
                num_samples += y.shape[0]
            correct = np.sum(np.array(ys) == np.array(y_pred))

            return {
                f'{target_data_split_name}_loss': total_loss,
                f'{target_data_split_name}_total': num_samples,
                f'{target_data_split_name}_avg_loss': total_loss /
                float(num_samples),
                f'{target_data_split_name}_acc': correct / float(num_samples)
            }

    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)
        return self.get_model_para()

    def get_model_para(self):
        return self.model.cpu().state_dict()

    def change_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr


def call_locoprop_trainer(trainer_type):
    if trainer_type == 'locoprop_trainer':
        return LocoPropTrainer


register_trainer('locoprop_trainer', call_locoprop_trainer)
