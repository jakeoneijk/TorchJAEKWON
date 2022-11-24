from typing import Optional

import os
import importlib

class GetModule:
    def __init__(self) -> None:
        self.root_path_dict:dict[str,str] = dict()
        self.root_path_dict["preprocess"] = "./DataProcess/Preprocess"
        self.root_path_dict["make_meta_data"] = "./DataProcess/MakeMetaData"
        self.root_path_dict["pytorch_dataLoader"] = "./Data/PytorchDataLoader"
        self.root_path_dict["pytorch_dataset"] = "./Data/PytorchDataset"
        self.root_path_dict["batch_sampler"] = "./Data/PytorchDataLoader/BatchSampler"
        self.root_path_dict["process"] = "./DataProcess/Process"
        self.root_path_dict["log_writer"] = "./Train/LogWriter"
        self.root_path_dict["trainer"] = "./Train/Trainer"
        self.root_path_dict["model"] = "./Model"
        self.root_path_dict["optimizer"] = "./Train/Optimizer"
        self.root_path_dict["loss_control"] = "./Train/Loss/LossControl"
        self.root_path_dict["loss_function"] = "./Train/Loss/LossFunction"
        self.root_path_dict["tester"] = "./Test/Tester"
        self.root_path_dict["evaluater"] = "./Evaluater"

        self.preprocess_realtime_root_path:str = "./PreprocessRealTime"
        self.trainer_root_path:str = "./Train/Trainer"
    
    def get_import_path_of_module(self,root_path:str, module_name:str ) -> Optional[str]:
        path_queue:list = [root_path, root_path.replace("./","./TorchJAEKWON/")]
        while path_queue:
            path_to_search:str = path_queue.pop(0)
            for dir_name in os.listdir(path_to_search):
                path:str = path_to_search + "/" + dir_name
                if os.path.isdir(path):
                    path_queue.append(path)
                else:
                    file_name:str =  os.path.splitext(dir_name)[0]
                    if file_name == module_name:
                        final_path:str = (path_to_search + "/" + file_name).replace("./","").replace("/",".")
                        return final_path
        return None
    
    def get_module(self,module_type,module_name,module_arg,arg_unpack=False) -> object:
        module_path:str = self.get_import_path_of_module(self.root_path_dict[module_type],module_name)
        module_from = importlib.import_module(module_path)
        module_import_class = getattr(module_from,module_name)
        if module_arg is not None:
            module = module_import_class(**module_arg) if arg_unpack else module_import_class(module_arg)
        else:
            module = module_import_class()
        
        return module
    
    def get_model(
        self,
        model_name:str
        ):
        
        module_file_path:str = self.get_import_path_of_module(self.root_path_dict["model"], model_name)
        file_module = importlib.import_module(module_file_path)
        class_module = getattr(file_module,model_name)
        model_parameter:dict = class_module.get_argument_of_this_model()
        model = class_module(**model_parameter)
        return model