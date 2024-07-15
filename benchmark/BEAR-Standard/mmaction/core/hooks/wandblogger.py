import torch
from mmcv.runner.hooks import HOOKS, LoggerHook
from collections import OrderedDict
import wandb

@HOOKS.register_module()
class WandbHook(LoggerHook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=1, name='mpii-tsh', project='tadvar', group='marco'):
        super().__init__(interval=interval)
        # self.log_dict = {}
        wandb.init(name=name, project=project, group=group)

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        log_dict = dict(log_dict, **runner.log_buffer.output)

    # def after_train_iter(self, runner):
    #     # accumulate loss, accuracy, etc for each iteration
    #     outputs = runner.outputs['log_vars']
    #     for key, val in outputs.items():
    #         key = 'train_' + key
    #         if key not in self.log_dict:
    #             self.log_dict[key] = []
    #         self.log_dict[key].append(val)

    # def after_train_epoch(self, runner):
    #     # sum total loss and average accuracy
    #     for key, val in self.log_dict:
    #         if 'acc' in key:
    #             self.log_dict[key] = sum(val) / len(val)
    #         if 'loss' in key:
    #             self.log_dict[key] = sum(val)
    #     # log the gradients and params
    #     model = runner.model
    #     self.log_dict['params'] = {}
    #     self.log_dict['grads'] = {}
    #     for name, param in model.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             self.log_dict['params'][name] = param.cpu()
    #             self.log_dict['grads'][name] = param.grad.cpu()
    #     print(self.log_dict)
    #     wandb.log(self.log_dict)
    #     self.log_dict = {}

    # def after_val_iter(self, runner):
    #     # accumulate loss and accuracy after each iter
    #     outputs = runner.outputs['log_vars']
    #     for key, val in outputs.items():
    #         key = 'val_' + key
    #         if key not in self.log_dict:
    #             self.log_dict[key] = []
    #         self.log_dict[key].append(val)

    # def after_val_epoch(self, runner):
    #     # sum losses and average accuracy
    #     for key, val in self.log_dict:
    #         if 'acc' in key:
    #             self.log_dict[key] = sum(val) / len(val)
    #         if 'loss' in key:
    #             self.log_dict[key] = sum(val)
    #     print(self.log_dict)
    #     wandb.log(self.log_dict)
    #     self.log_dict = {}
