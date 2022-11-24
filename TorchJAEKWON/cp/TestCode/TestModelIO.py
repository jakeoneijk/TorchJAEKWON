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
    def __init__(self,h_params:HParams) -> None:
        get_module : GetModule = GetModule(h_params)
        self.test_model : nn.Module = get_module.get_model()
    
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
    h_params : HParams = HParams()
    test_module : TestModelIO = TestModelIO(h_params)
    test_module.test(False)






"""
import sys
sys.path.append("..")

import torch
import torch.nn as nn

import time
from DataProcess.Util.UtilAudio import UtilAudio
from DataProcess.Util.UtilAudioSTFT import UtilAudioSTFT

from HParams import HParams

from Data.PytorchDataLoader.PytorchDataLoader import PytorchDataLoader
from Train.LossFunction.LossControl import LossControl


class TestModelIO():
    def __init__(self,h_params:HParams):
        self.get_module = GetModule(h_params)
        self.test_model: nn.Module = self.get_module.get_model({"h_params":h_params})
    
    def get_param_num(self, model:nn.Module):
        num_param:int = sum(param.numel() for param in model.parameters())
        return num_param

    def test(self):
        state_dict_debug = self.test_model.state_dict()
        model_input = self.test_model.get_input_size()
        summary(self.test_model, model_input, device="cpu")
        print(self.get_param_num(self.test_model))

class TestPytorchDataLoader:
    def __init__(self, h_params:HParams) -> None:
        self.pytorch_data_loader = PytorchDataLoader(h_params)
    
    def test(self):
        dara_loaders_dict = self.pytorch_data_loader.get_pytorch_data_loaders()
        current_time = time.time()
        print("start")
        for i,data in enumerate(dara_loaders_dict["train"]): 
            print(f"{i+1}")
            if (i+1) == 5000:
                break
        print(f"time for 100 iterations: {time.time() - current_time}")

class TestLossControl:
    def __init__(self, h_params:HParams) -> None:
        self.loss_control = LossControl(h_params)
    
    def test(self):
        pass

class TestLatentFeature:
    def __init__(self, h_params:HParams) -> None:
        self.h_params = h_params
        self.get_module = GetModule(h_params)
        self.test_model: nn.Module = self.get_module.get_model({"h_params":h_params})
        if self.h_params.process.name is not None:
            self.data_processor = self.get_module.get_module("process",self.h_params.process.name, {"h_params":self.h_params},arg_unpack=True)
        else:
            self.data_processor = None

        self.util_stft = UtilAudioSTFT(h_params)
        self.util_audio = UtilAudio()
        self.device = h_params.resource.device
    
    def load_model(self):
        import torch
        load_data= torch.load("/home/jakeoneijk/210705_NeuralVocoderSeparation/Train/Log/220426_mel_gen_emb_revaug/train_checkpoint1.pth")
        self.test_model.load_state_dict(load_data['models'])
        self.test_model = self.test_model.to(self.device)
    
    def get_emb(self,input_feature):
        final_shape_len = 3
        input_feature = torch.from_numpy(input_feature).float()
        for _ in range(final_shape_len - len(input_feature.shape)):
            input_feature = torch.unsqueeze(input_feature, 0)
        input_feature = input_feature.to(self.device)

        input_feature = self.data_processor.preprocess_input_for_inference(input_feature)
        with torch.no_grad():
            pred_features_torch = self.test_model(input_feature)
        
        emb= pred_features_torch["embedding_center"]
        return emb


    def test(self):
        self.load_model()
        input_feature = self.util_audio.read_audio_fix_channels_of_mono_stereo("./gt_vocal1.wav",sample_rate=self.h_params.preprocess.sample_rate, mono=False, read_type = "soundfile")
        input_feature = input_feature.T
        cut = 70000
        input_dict = dict()
        input_dict["type1_1"] = input_feature[...,cut*3:cut*3+cut]
        input_dict["type1_2"] = input_feature[...,cut*2:cut*2+cut]

        

        input_feature2 = self.util_audio.read_audio_fix_channels_of_mono_stereo("./gt_vocal1.wav",sample_rate=self.h_params.preprocess.sample_rate, mono=False, read_type = "soundfile")
        input_feature2 = input_feature2.T
        input_dict["type2_1"] = input_feature2[...,cut*3:cut*3+cut]
        input_dict["type2_2"] = input_feature2[...,cut*2:cut*2+cut]

        emb_dict = dict()
        for key in input_dict:
            emb_dict[key] = self.get_emb(input_dict[key])

        print("wow")

    


if __name__ == "__main__":
    
    #h_params.mode.debug_mode = True
    #test_module = TestPytorchDataLoader(h_params)
    test_module = TestModelIO(h_params)
    #test_module = TestLatentFeature(h_params)
    test_module.test()


"""
