# Local Learning Abstraction: Trainer

FederatedScope decouples the local learning process and details of FL communication and schedule, allowing users to freely customize the local learning algorithm via the `trainer`. Each worker holds a `trainer` object to manage the details of local learning, such as the loss function, optimizer, training step, evaluation, etc.

This tutorial is a shorter version of [Doc](https://federatedscope.io/docs/trainer/), where you can access more details in the full version tutorials.

## Code Structure

The code structure is shown below, and we will discuss all the concepts of our FS Trainer later.

```bash
federatedscope/core
├── trainers
│   ├── BaseTrainer
│   │   ├── Trainer
│   │   │   ├── GeneralTorchTrainer
│   │   │   ├── GeneralTFTrainer
│   │   │   ├── Context
│   │   │   ├── ...
│   │   ├── UserDefineTrainer
│   │   ├── ...
```

## FS Trainer

A typical machine-learning process consists of the following procedures:

1. Preparing data.
2. Iterations over training datasets to update the model parameters
3. Evaluation of the quality of the learned model on validation/evaluation datasets
4. Saving, loading, and monitoring the model and intermediate results

### BaseTrainer

`BaseTrainer` is an abstract class of our Trainer, which provide the interface of each method. And you can implement your own trainer by inheriting from `BaseTrainer`. More examples can be found in `federatedscope/contrib/trainer`.

```python
class BaseTrainer(abc.ABC):
    def __init__(self, model, data, device, **kwargs):
        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, target_data_split_name='test'):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model_parameters, strict=False):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_para(self):
        raise NotImplementedError
    
    ... ...
```

### Trainer

As the figure shows, in FederatedScope `Trainer` (a subclass of `BaseTrainer`), these above procedures are provided with high-level `routines` abstraction, which is made up of `Context` class and several pluggable `Hooks`. And we provide `GeneralTorchTrainer` and `GeneralTFTrainer` for `PyTorch` and `TensorFlow`, separately.

<img src="https://img.alicdn.com/imgextra/i4/O1CN01H8OEeS1tdhR38C4dK_!!6000000005925-2-tps-1504-874.png" alt="undefined" style="zoom:50%;" />

#### Context
TBD
The `Context` class is used to hold learning-related attributes, including data, model, optimizer and etc.

```python
dict_keys(['train_data', 'train_loader', 'num_train_data', 'val_data', 'val_loader', 'num_val_data', 'test_data', 'test_loader', 'num_test_data', 'lifecycles', 'cfg', 'model', 'data', 'device', 'cur_mode', 'mode_stack', 'cur_split', 'split_stack', 'trainable_para_names', 'criterion', 'regularizer', 'grad_clip', 'num_train_batch', 'num_train_batch_last_epoch', 'num_train_epoch', 'num_total_train_batch', 'num_val_epoch', 'num_val_batch', 'num_test_epoch', 'num_test_batch', 'monitor', 'models', 'mirrored_models', 'optimizer', 'scheduler', 'loss_batch_total', 'loss_regular_total', 'num_samples', 'ys_true', 'ys_prob', 'eval_metrics'])
```

#### Hooks

The `Hooks` represent fine-grained learning behaviors at different point-in-times, which provides a simple yet powerful way to customize learning behaviors with a few modifications and easy re-use of fruitful default hooks. In this section, we will show the detail of each hook in `Trainer`.

* hook1
* hook2
* hook3