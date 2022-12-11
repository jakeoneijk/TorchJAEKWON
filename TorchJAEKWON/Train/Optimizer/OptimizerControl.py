import torch
import torch.nn as nn

from HParams import HParams

class OptimizerControl:
    def __init__(self, model:nn.Module = None) -> None:
        self.h_params = HParams()
        self.optimizer = None
        self.lr_scheduler = None

        self.scheduler_config = self.h_params.train.scheduler
        self.num_lr_scheduler_step = 0

        if model is not None:
            self.set_optimizer(model)
            self.set_lr_scheduler()
    
    def set_optimizer(self,model:nn.Module):
        optimizer_name = self.h_params.train.optimizer["name"]

        optimizer_config = self.h_params.train.optimizer["config"]
        optimizer_config["params"] = model.parameters()
        optimizer_config['lr'] = float(optimizer_config['lr'])
        if 'eps' in optimizer_config:
            optimizer_config['eps'] = float(optimizer_config['eps'])

        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(**optimizer_config)
    
    def optimizer_step(self):
        self.optimizer.step()
    
    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
    
    def optimizer_state_dict(self):
        return self.optimizer.state_dict()
    
    def optimizer_load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def lr_scheduler_state_dict(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler.state_dict()
    
    def lr_scheduler_load_state_dict(self, state_dict):
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict)
    
    def set_lr_scheduler(self):
        pass

    def lr_scheduler_step(self,interval_type="step",args = None):
        if self.lr_scheduler == None or (self.num_lr_scheduler_step % self.scheduler_config["config"]["frequency"]) != 0 or interval_type != self.scheduler_config["config"]["interval"]:
            return 

        self.lr_scheduler.step()
        self.num_lr_scheduler_step += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]