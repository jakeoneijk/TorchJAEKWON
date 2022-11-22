import torch
import torch.nn as nn
from functools import partial

from HParams import HParams

from Train.Optimizer.OptimizerControl import OptimizerControl
from torch.optim.lr_scheduler import LambdaLR

class OptimizerControlLambdaLRByStep(OptimizerControl):
    def __init__(self,h_params:HParams, model:nn.Module = None) -> None:
        super().__init__(h_params, model)
    
    def set_lr_scheduler(self):
        scheduler_config = self.h_params.train.scheduler["config"]
        self.lr_scheduler = LambdaLR(
            self.optimizer, 
            partial( get_lr_lambda, warm_up_steps=scheduler_config["warm_up_steps"], reduce_lr_steps=scheduler_config["reduce_lr_steps"])
            )
   
def get_lr_lambda(step, warm_up_steps: int, reduce_lr_steps: int):
    r"""Get lr_lambda for LambdaLR. E.g.,
    .. code-block: python
        lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)
        from torch.optim.lr_scheduler import LambdaLR
        LambdaLR(optimizer, lr_lambda)
    Args:
        warm_up_steps: int, steps for warm up
        reduce_lr_steps: int, reduce learning rate by 0.9 every #reduce_lr_steps steps
    Returns:
        learning rate: float
    """
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 0.9 ** (step // reduce_lr_steps)