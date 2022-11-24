from typing import Union
from numpy import ndarray
from torch import Tensor

import os
import pickle
import yaml
import csv
from pathlib import Path

class UtilData:

    def get_file_name_from_path(self,path:str,with_ext:bool)->str:
        path_pathlib = Path(path)
        if with_ext:
            return path_pathlib.name
        else:
            return path_pathlib.stem
    
    def pickle_save(self,save_path:str, data:Union[ndarray,Tensor]) -> None:
        assert(os.path.splitext(save_path)[1] == ".pkl") , "file extension should be '.pkl'"

        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        
        with open(save_path,'wb') as file_writer:
            pickle.dump(data,file_writer)
    
    def pickle_load(self,data_path:str) -> Union[ndarray,Tensor]:
        with open(data_path, 'rb') as pickle_file:
            data:Union[ndarray,Tensor] = pickle.load(pickle_file)
        return data
    
    def yaml_save(self,save_path:str, data:Union[dict,list]) -> None:
        assert(os.path.splitext(save_path)[1] == ".yaml") , "file extension should be '.yaml'"

        with open(save_path, 'w') as file:
            yaml.dump(data, file)
    
    def yaml_load(self,data_path:str) -> dict:
        yaml_file = open(data_path, 'r')
        return yaml.safe_load(yaml_file)
    
    def csv_load(self,data_path:str) -> list:
        row_result_list = list()
        with open(data_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            for row in spamreader:
                row_result_list.append(row)
        return row_result_list