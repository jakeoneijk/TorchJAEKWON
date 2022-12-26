from typing import Dict

import os
from abc import ABC, abstractmethod
from enum import Enum,unique
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from HParams import HParams
from TorchJAEKWON.GetModule import GetModule
from TorchJAEKWON.Data.PytorchDataLoader.PytorchDataLoader import PytorchDataLoader
from TorchJAEKWON.Train.LogWriter.LogWriter import LogWriter
from TorchJAEKWON.Train.Optimizer.OptimizerControl import OptimizerControl
from TorchJAEKWON.Train.AverageMeter import AverageMeter
from TorchJAEKWON.Train.Loss.LossControl.LossControl import LossControl

@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer(ABC):
    def __init__(self) -> None:
        self.h_params = HParams()

        self.model = None
        
        self.get_module:GetModule = GetModule()
        self.optimizer_control:OptimizerControl = None
        self.loss_control: LossControl = None

        self.train_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None

        self.seed = (int)(torch.cuda.initial_seed() / (2**32)) if self.h_params.train.seed is None else self.h_params.train.seed
        self.set_seeds(self.h_params.train.seed_strict)

        self.current_epoch:int = 0
        self.total_epoch:int = self.h_params.train.epoch

        self.best_valid_metric:dict[str,AverageMeter] = None
        self.best_valid_epoch:int = 0

        self.global_step:int = 0
        self.local_step:int = 0

        self.log_writer:LogWriter = None

    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    
    @abstractmethod
    def run_step(self,data,metric,train_state:TrainState):
        """
        run 1 step
        1. get data
        2. use model

        3. calculate loss
            current_loss_dict = self.loss_control.calculate_total_loss_by_loss_meta_dict(pred_dict=pred, target_dict=train_data_dict)
        
        4. put the loss in metric (append)
            for loss_name in current_loss_dict:
                metric[loss_name].update(current_loss_dict[loss_name].item(),batch_size)

        return current_loss_dict["total_loss"],metric
        """
        raise NotImplementedError

    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        if prev_best_metric is None:
            return current_metric
        
        if prev_best_metric[self.loss_control.final_loss_name].avg > current_metric[self.loss_control.final_loss_name].avg:
            self.save_module()
            self.best_valid_epoch = self.current_epoch
            return current_metric
        else:
            return prev_best_metric
    
    def log_metric(
        self, 
        metrics:Dict[str,AverageMeter],
        data_size: int,
        train_state=TrainState.TRAIN
        )->None:
        """
        log and visualizer log
        """
        if train_state == TrainState.TRAIN:
            x_axis_name:str = "step_global"
            x_axis_value:int = self.global_step
        else:
            x_axis_name:str = "epoch"
            x_axis_value:int = self.current_epoch

        log:str = f'Epoch ({train_state.value}): {self.current_epoch:03} ({self.local_step}/{data_size}) global_step: {self.global_step}\t'
        
        for metric_name in metrics:
            val:float = metrics[metric_name].avg
            log += f' {metric_name}: {val:.06f}'
            self.log_writer.visualizer_log(
                x_axis_name=x_axis_name,
                x_axis_value=x_axis_value,
                y_axis_name=f'{train_state.value}/{metric_name}',
                y_axis_value=val
            )
        self.log_writer.print_and_log(log,self.global_step)

    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def set_seeds(self,strict=False):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if strict:
                torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)

    def init_train(self, dataset_dict=None):
        self.model:nn.Module = self.get_module.get_model(self.h_params.model.class_name)
        self.optimizer_control = self.get_module.get_module("optimizer",self.h_params.train.optimizer_control_config['class_name'],{"model":self.model},arg_unpack=True)
        self.loss_control = self.get_module.get_module("loss_control",self.h_params.train.loss_control["class_name"],None)

        if self.h_params.resource.multi_gpu:
            from TorchJAEKWON.Train.Trainer.Parallel import DataParallelModel, DataParallelCriterion
            self.model = DataParallelModel(self.model)
            self.model.cuda()
            for loss_name in self.loss_control.loss_function_dict:
                self.loss_control.loss_function_dict[loss_name] = DataParallelCriterion(self.loss_control.loss_function_dict[loss_name])
        else:
            self.loss_control.to(self.h_params.resource.device)
            self.model = self.model.to(self.h_params.resource.device)

        self.log_writer = self.get_module.get_module(
            module_type="log_writer",
            module_name=self.h_params.train.log_writer_class_name,
            module_arg={"model":self.model})

        self.set_data_loader(dataset_dict)
    
    def set_data_loader(self,dataset_dict=None):
        data_loader_loader:PytorchDataLoader = self.get_module.get_module('pytorch_dataLoader',self.h_params.pytorch_data.class_name,None)

        if dataset_dict is not None:
            pytorch_data_loader_config_dict = data_loader_loader.get_pytorch_data_loader_config(dataset_dict)
            data_loader_dict = data_loader_loader.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        else:
            data_loader_dict = data_loader_loader.get_pytorch_data_loaders()

        self.train_data_loader = data_loader_dict["train"]
        self.valid_data_loader = data_loader_dict["valid"]
        self.test_data_loader = None
    
    def fit(self):
        for _ in range(self.current_epoch, self.total_epoch):
            self.log_writer.print_and_log(f'----------------------- Start epoch : {self.current_epoch} / {self.h_params.train.epoch} -----------------------',self.global_step)
            self.log_writer.print_and_log(f'current best epoch: {self.best_valid_epoch}',self.global_step)
            if self.best_valid_metric is not None:
                for loss_name in self.best_valid_metric:
                    self.log_writer.print_and_log(f'{loss_name}: {self.best_valid_metric[loss_name].avg}',self.global_step)
            
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------',self.global_step)
            self.log_writer.print_and_log(f'current lr: {self.optimizer_control.get_lr()}',self.global_step)
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------',self.global_step)
    
            #Train
            self.log_writer.print_and_log('train_start',self.global_step)
            self.run_epoch(self.train_data_loader,TrainState.TRAIN, metric_range = "step")
            
            #Valid
            self.log_writer.print_and_log('valid_start',self.global_step)

            with torch.no_grad():
                valid_metric = self.run_epoch(self.valid_data_loader,TrainState.VALIDATE, metric_range = "epoch")
                self.optimizer_control.lr_scheduler_step(interval_type="epoch", args=valid_metric)
            
            self.best_valid_metric = self.save_best_model(self.best_valid_metric, valid_metric)

            if self.current_epoch > self.h_params.train.save_model_after_epoch and self.current_epoch % self.h_params.train.save_model_every_epoch == 0:
                self.save_module(name=f"pretrained_epoch{str(self.current_epoch).zfill(8)}")
            
            self.current_epoch += 1
            self.log_writer.log_every_epoch(model=self.model)

        self.log_writer.print_and_log(f'best_epoch: {self.best_valid_epoch}',self.global_step)
        self.log_writer.print_and_log("Training complete",self.global_step)
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step"):
        assert metric_range in ["step","epoch"], "metric range should be 'step' or 'epoch'"

        if train_state == TrainState.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        dataset_size = len(dataloader)

        if metric_range == "epoch":
            metric = self.metric_init()

        for step,data in enumerate(dataloader):

            if metric_range == "step":
                metric = self.metric_init()

            if step >= len(dataloader):
                break

            self.local_step = step
            loss,metric = self.run_step(data,metric,train_state)
        
            if train_state == TrainState.TRAIN:
                self.optimizer_control.optimizer_zero_grad()
                loss.backward()
                self.optimizer_control.optimizer_step()

                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                
                self.global_step += 1

                self.optimizer_control.lr_scheduler_step(interval_type="step")
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)

        if train_state == TrainState.TRAIN:
            self.save_checkpoint()
            self.save_checkpoint("train_checkpoint_backup.pth")

        return metric
    
    def metric_init(self):
        loss_name_list = self.loss_control.get_loss_function_name_list()
        initialized_metric = dict()

        for loss_name in loss_name_list:
            initialized_metric[loss_name] = AverageMeter()

        return initialized_metric

    def save_module(self,name = 'pretrained_best_epoch'):
        path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
        torch.save(self.model.state_dict() if not self.h_params.resource.multi_gpu else self.model.module.state_dict(), path)

    def load_module(self,name = 'pretrained_best_epoch'):
        path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def save_checkpoint(self,save_name:str = 'train_checkpoint.pth'):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'models': self.model.state_dict() if not self.h_params.resource.multi_gpu else self.model.module.state_dict(),
            'optimizers': self.optimizer_control.optimizer_state_dict(),
            'lr_scheduler': self.optimizer_control.lr_scheduler_state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }
        path = os.path.join(self.log_writer.log_path["root"],save_name)
        self.log_writer.print_and_log(save_name,self.global_step)
        torch.save(train_state,path)

    def load_train(self,filename:str):
        cpt = torch.load(filename)
        self.seed = cpt['seed']
        self.set_seeds(self.h_params.train.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']
        self.model.load_state_dict(cpt['models'])
        self.optimizer_control.optimizer_load_state_dict(cpt['optimizers'])
        self.optimizer_control.lr_scheduler_load_state_dict(cpt['lr_scheduler'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']