import matplotlib.pyplot as plt
import numpy as np
import os
from federatedscope.core.message import Message
import logging

logger = logging.getLogger(__name__)


def plot_target_loss(loss_list, outdir):
    '''

    Args:
        loss_list: the list of loss regrading the target data
        outdir: the directory to store the loss

    '''

    target_data_loss = np.vstack(loss_list)
    logger.info(target_data_loss.shape)
    plt.plot(target_data_loss)
    plt.savefig(os.path.join(outdir, 'target_loss.png'))
    plt.close()


def sav_target_loss(loss_list, outdir):
    """
    Save the target loss in a text file

    Args:
        loss_list: write your description
        outdir: write your description
    """
    target_data_loss = np.vstack(loss_list)
    np.savetxt(os.path.join(outdir, 'target_loss.txt'),
               target_data_loss.transpose(),
               delimiter=',')


def callback_funcs_for_finish(self, message: Message):
    """
    Callback functions for finishing the training.

    Args:
        self: write your description
        message: write your description
    """
    logger.info("============== receiving Finish Message ==============")
    if message.content is not None:
        self.trainer.update(message.content)
    if self.is_attacker and self._cfg.attack.attack_method.lower(
    ) == "gradascent":
        logger.info(
            "============== start attack post-processing ==============")
        plot_target_loss(self.trainer.ctx.target_data_loss,
                         self.trainer.ctx.outdir)
        sav_target_loss(self.trainer.ctx.target_data_loss,
                        self.trainer.ctx.outdir)


def add_atk_method_to_Client_GradAscent(client_class):
    """
    Add methods to Client class that calls the finish method for each gradient descent iteration.

    Args:
        client_class: write your description
    """

    setattr(client_class, 'callback_funcs_for_finish',
            callback_funcs_for_finish)
    return client_class
