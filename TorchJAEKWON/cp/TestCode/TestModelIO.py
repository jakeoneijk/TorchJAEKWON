import sys
import os
from torch import Tensor

dir_of_current_file_is_located : str = os.path.dirname(__file__)
abs_dir_of_current_file_is_located : str = os.path.abspath(dir_of_current_file_is_located)
parent_dir : str = os.path.dirname(abs_dir_of_current_file_is_located)
sys.path.append(parent_dir)

import torch.nn as nn
from TestCode.torchinfo import summary

from HParams import HParams
from TorchJAEKWON.GetModule import GetModule

class TestModelIO():
    def __init__(self) -> None:
        get_module : GetModule = GetModule()
        h_params = HParams()
        self.test_model : nn.Module = get_module.get_model(h_params.model.class_name)
    
    def get_total_param_num(self, model:nn.Module) -> int:
        num_param : int = sum(param.numel() for param in model.parameters())
        return num_param
    
    def get_trainable_param_num(self, model:nn.Module) -> int:
        trainable_param : int = sum(param.numel() for param in model.parameters() if param.requires_grad)
        return trainable_param
    
    def pretty_print_int(self,number : int) -> str:
        string_pretty_int : str = format(number, ',d')
        return string_pretty_int

    def test(self,use_torchinfo:bool = False) -> None:
        if use_torchinfo:
            model_input:Tensor = self.test_model.get_test_input()
            summary(model = self.test_model, x = model_input, device="cpu")
        else:
            result_param_dict : dict = dict()
            result_param_dict["Total params"] = self.pretty_print_int(self.get_total_param_num(self.test_model))
            result_param_dict["Trainable params"] = self.pretty_print_int(self.get_trainable_param_num(self.test_model))
            
            for key in result_param_dict:
                print(f'{key} : {result_param_dict[key]}')
            
            model_input:Tensor = self.test_model.get_test_input()
            self.test_model(model_input)

if __name__ == "__main__":
    test_module : TestModelIO = TestModelIO()
    test_module.test(use_torchinfo = False)