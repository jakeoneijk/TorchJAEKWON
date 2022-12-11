import os
from torch.utils.data import DataLoader

from HParams import HParams
from TorchJAEKWON.GetModule import GetModule

class PytorchDataLoader:
    def __init__(self):
        self.h_params = HParams()
        self.data_loader_config:dict = self.h_params.pytorch_data.dataloader
        self.get_module = GetModule()
        self.module_name_list_of_data_loader_args:list = ["batch_sampler"]
    
    def get_pytorch_data_loaders(self) -> dict:
        data_path_dict:dict = self.get_data_path_dict()
        pytorch_dataset_dict = self.get_pytorch_data_set_dict(data_path_dict)
        pytorch_data_loader_config_dict = self.get_pytorch_data_loader_config(pytorch_dataset_dict)
        pytorch_data_loader_dict = self.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        return pytorch_data_loader_dict

    def get_data_path_dict(self) -> dict :
        data_path_dict:dict = {subset:[] for subset in self.data_loader_config}
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            if self.h_params.data.data_config_per_dataset_dict[data_name]["load_to_pytorch_dataset"]:
                data_path:str = f"{self.h_params.data.root_path}/{data_name}"
                for subset in data_path_dict:
                    data_path_dict[subset] += [f"{data_path}/{subset}/{fname}" for fname in os.listdir(f"{data_path}/{subset}")]
        
        if self.h_params.mode.debug_mode:
            print("use small data because of debug mode")
            for subset in data_path_dict:
                data_path_dict[subset] = data_path_dict[subset][:self.data_loader_config[subset]["batch_size"]]
        
        return data_path_dict
    
    def get_pytorch_data_set_dict(self,data_path_dict) -> dict:
        pytorch_dataset_dict = dict()
        for subset in self.data_loader_config:
            config_for_dataset = {
                "data_path_list": data_path_dict[subset],
                "subset": subset
            }
            pytorch_dataset_dict[subset] = self.get_module.get_module("pytorch_dataset",self.data_loader_config[subset]["dataset"]["class_name"],config_for_dataset)
        return pytorch_dataset_dict
    
    def get_pytorch_data_loader_config(self,pytorch_dataset:dict) -> dict:
        pytorch_data_loader_config_dict = dict()

        for subset in pytorch_dataset:
            args_exception_list = self.get_exception_list_of_dataloader_args_config(subset)
            pytorch_data_loader_config_dict[subset] = dict()
            pytorch_data_loader_config_dict[subset]["dataset"] = pytorch_dataset[subset]
            for arg_name in self.data_loader_config[subset]:
                if arg_name in args_exception_list:
                    continue
                if arg_name in self.module_name_list_of_data_loader_args:
                    arguments_for_args_class = {"h_params":self.h_params,"config":self.data_loader_config[subset][arg_name],"subset":subset}
                    pytorch_data_loader_config_dict[subset][arg_name] = self.get_module.get_module( arg_name, 
                                                                                                    self.data_loader_config[subset][arg_name]["class_name"],
                                                                                                    arguments_for_args_class)
                else:
                    pytorch_data_loader_config_dict[subset][arg_name] = self.data_loader_config[subset][arg_name]
        
        return pytorch_data_loader_config_dict
    
    def get_exception_list_of_dataloader_args_config(self,subset):
        args_exception_list = ["dataset"]
        if "batch_sampler" in self.data_loader_config[subset]:
            args_exception_list += ["batch_size", "shuffle", "sampler", "drop_last"]
        return args_exception_list

    def get_pytorch_data_loaders_from_config(self,dataloader_config:dict) -> dict:
        pytorch_data_loader_dict = dict()
        for subset in dataloader_config:
            pytorch_data_loader_dict[subset] = DataLoader(**dataloader_config[subset])
        return pytorch_data_loader_dict



    