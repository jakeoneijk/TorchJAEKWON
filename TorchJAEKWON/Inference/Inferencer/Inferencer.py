from typing import List,Union
from torch import Tensor



import os
import torch
import torch.nn as nn

from HParams import HParams
from TorchJAEKWON.GetModule import GetModule
from TorchJAEKWON.DataProcess.Util.UtilData import UtilData

class Inferencer():
    def __init__(self) -> None:
        self.h_params = HParams()
        self.get_module = GetModule()
        self.util_data = UtilData()

        self.model:Union[nn.Module,object] = self.get_model()
        self.output_dir:str = None
    
    def get_model(self) -> Union[nn.Module,object]:
        return self.get_module.get_model(self.h_params.model.class_name)
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    
    def get_testset_meta_data_list(self) -> List[dict]:
        meta_data_list = list()
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            meta:list = self.util_data.pickle_load(f'{self.h_params.data.root_path}/{data_name}_test.pkl')
            meta_data_list += meta
        return meta_data_list

    def set_output_dir_path_by_pretrained_name_and_meta_data(self,pretrained_name:str,meta_data:dict):
        self.output_dir:str = f"{self.h_params.inference.output_dir}/{self.h_params.inference.pretrain_dir}({pretrained_name})/{meta_data['name']}"
    
    def read_data_dict_by_meta_data(self,meta_data:dict)->dict:
        '''
        {
            "model_input":
            "gt": {
                "audio",
                "spectrogram"
            }
        }
        '''
        data_dict = dict()
        data_dict["gt"] = dict()
        data_dict["pred"] = dict()
    
    def post_process(self,data_dict:dict)->dict:
        return data_dict

    def save_data(self,data_dict:dict):
        pass
    
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def inference(self) -> None:
        pretrained_path_list:list = self.get_pretrained_path_list()

        for pretrained_path in pretrained_path_list:
            self.pretrained_load(pretrained_path) 
            pretrained_name:str = self.util_data.get_file_name_from_path(pretrained_path,False)

            if self.h_params.inference.dataset_type == "onedata":
                print("not implemented yet")

            elif self.h_params.inference.dataset_type == "testset":

                meta_data_list:list = self.get_testset_meta_data_list()
                for i,meta_data in enumerate(meta_data_list):
                    print(f"{i+1}/{len(meta_data_list)}")
                    self.set_output_dir_path_by_pretrained_name_and_meta_data(pretrained_name,meta_data)
                    
                    if os.path.isdir(self.output_dir):
                        print(f"[{self.output_dir}] already exist!!")
                        continue
                    os.makedirs(self.output_dir,exist_ok=True)

                    data_dict:dict = self.read_data_dict_by_meta_data(meta_data=meta_data)
                    data_dict = self.update_data_dict_by_model_inference(data_dict)
                    
                    data_dict:dict = self.post_process(data_dict)
                    self.save_data(data_dict)
    
    def update_data_dict_by_model_inference(self,data_dict):
        if type(data_dict["model_input"]) == Tensor:
            with torch.no_grad():
                data_dict["pred"] = self.model(data_dict["model_input"].to(self.h_params.resource.device))
        return data_dict
                    

    
    def get_pretrained_path_list(self) -> list:
        pretrained_dir_path:str = f"{self.h_params.inference.pretrain_root_dir}/{self.h_params.inference.pretrain_dir}"
        
        if self.h_params.inference.pretrain_module_name in ["all","last_epoch"]:
            pretrain_name_list:list = [  pretrain_module 
                                    for pretrain_module in os.listdir(pretrained_dir_path)
                                    if pretrain_module.endswith("pth") and "checkpoint" not in pretrain_module]
            pretrain_name_list.sort()

        if self.h_params.inference.pretrain_module_name == "last_epoch":
            pretrain_name_list = [pretrain_name_list[-1]]
        
        return [f"{pretrained_dir_path}/{pretrain_name}" for pretrain_name in pretrain_name_list]
    
    def pretrained_load(self,pretrain_path:str) -> None:
        if pretrain_path is None:
            return
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(pretrained_load)
        self.model = self.model.to(self.h_params.resource.device)