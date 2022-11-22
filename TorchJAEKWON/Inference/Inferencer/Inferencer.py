from typing import List,Union
from torch import Tensor
import torch.nn as nn

import os
import torch
from abc import ABC, abstractmethod

from HParams import HParams
from GetModule import GetModule
from DataProcess.Util.UtilData import UtilData

class Inferencer(ABC):
    def __init__(self,h_params:HParams) -> None:
        self.h_params:HParams = h_params
        self.get_module = GetModule()
        self.util_data = UtilData()

        self.model:nn.Module = self.get_module.get_model(self.h_params.model.name)
        self.output_dir_path:str = None
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    @abstractmethod
    def get_testset_meta_data_list(self) -> List[dict]:
        pass

    def set_output_dir_path_by_pretrained_name_and_meta_data(self,pretrained_name:str,meta_data:dict):
        self.output_dir_path:str = f"{self.h_params.test.output_path}/{self.h_params.test.pretrain_dir_name}({pretrained_name})/{meta_data['name']}"
    
    @abstractmethod
    def read_data_dict_by_meta_data(self,meta_data:dict)->dict:
        '''
        {
            "model_input":
            "gt": {}
        }
        '''
        pass

    def post_process(self,data_dict:dict, model_output:Union[Tensor,dict])->dict:
        data_dict["pred"]["model_output"] = model_output.squeeze().detach().cpu().numpy()
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

            if self.h_params.test.dataset_type == "onedata":
                print("not implemented yet")

            elif self.h_params.test.dataset_type == "testset":

                meta_data_list:list = self.get_testset_meta_data_list()
                for i,meta_data in enumerate(meta_data_list):
                    #if meta_data["song_name"] != "Al James - Schoolboy Facination":
                    #    continue
                    print(f"{i+1}/{len(meta_data_list)}")
                    self.set_output_dir_path_by_pretrained_name_and_meta_data(pretrained_name,meta_data)
                    if os.path.isdir(self.output_dir_path):
                        print(f"[{self.output_dir_path}] already exist!!")
                        continue
                    os.makedirs(self.output_dir_path,exist_ok=True)

                    data_dict:dict = self.read_data_dict_by_meta_data(meta_data=meta_data)

                    with torch.no_grad():
                        pred:dict = self.model(data_dict["model_input"].to(self.h_params.resource.device))
                    
                    post_process_dict:dict = self.post_process(data_dict,pred)
                    self.save_data(post_process_dict)
                    

    
    def get_pretrained_path_list(self) -> list:
        pretrained_dir_path:str = f"{self.h_params.test.pretrain_path}/{self.h_params.test.pretrain_dir_name}"
        
        if self.h_params.test.pretrain_module_name == "all":
            pretrain_name_list:list = [  pretrain_module 
                                    for pretrain_module in os.listdir(pretrained_dir_path)
                                    if pretrain_module.endswith("pth") and "checkpoint" not in pretrain_module]
        
        return [f"{pretrained_dir_path}/{pretrain_name}" for pretrain_name in pretrain_name_list]
    
    def pretrained_load(self,pretrain_path:str) -> None:
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(pretrained_load)
        self.model.to(self.h_params.resource.device)