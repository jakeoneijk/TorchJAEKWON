from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import os
import time
from tkinter.messagebox import NO

from HParams import HParams

class Preprocessor(ABC):
    def __init__(self,h_params:HParams, data_config_dict:dict = None) -> None:
        self.h_params: HParams = h_params
        self.preprocessed_data_path:str = ""
        self.data_name:str = None
    
    def write_message(self,message_type,message) -> None:
        file = open(f"{self.preprocessed_data_path}/{message_type}.txt",'a')
        file.write(message+'\n')
        file.close()
    
    def preprocess_data(self) -> None:
        self.set_preprocessed_data_path()
        meta_param_list:list = self.get_meta_data_param()
        start_time = time.time()

        if self.h_params.preprocess.multi_processing:
            with ProcessPoolExecutor(max_workers=None) as pool:
                pool.map(self.preprocess_one_data, meta_param_list)
        else:
            for i,meta_param in enumerate(meta_param_list):
                print(f"{i+1}/{len(meta_param_list)}")
                self.preprocess_one_data(meta_param)


        print("Finish preprocess. {:.3f} s".format(time.time() - start_time))
    
    def set_preprocessed_data_path(self) -> None:
        assert self.data_name != None, "you should set self.data_name first"
        self.preprocessed_data_path = os.path.join(self.h_params.data.root_path,self.data_name)

    @abstractmethod
    def get_meta_data_param(self) -> list:
        raise NotImplementedError
    
    @abstractmethod
    def preprocess_one_data(self,param: tuple) -> None:
        raise NotImplementedError
        