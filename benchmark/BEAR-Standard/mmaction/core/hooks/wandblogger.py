import torch
from mmcv.runner.hooks import HOOKS, Hook
import wandb

@HOOKS.register_module()
class WandbHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=1, name='mpii-tsh', project='tadvar', group='marco'):
        self.interval = interval
        wandb.init(name=name, project=project, group=group)

    def after_val_epoch(self, runner):
        log_dict = {'val_loss': runner.outputs['loss']}
        log_dict.update(runner.outputs['log_vars'])
        wandb.log(log_dict)
