from typing import Dict
from torch import Tensor

from HParams import HParams
from DataProcess.Process.Process import Process
from DataProcess.Util.UtilAudioMelSpec import UtilAudioMelSpec
from DataProcess.Util.UtilAudioVocalPresence import UtilAudioVocalPresence


class ProcessMelSpectrogram(Process):
    def __init__(self,h_params:HParams) -> None:
        super(ProcessMelSpectrogram,self).__init__(h_params)
        self.util:UtilAudioMelSpec = UtilAudioMelSpec(self.h_params)
        self.util_vocal_presence = UtilAudioVocalPresence()
        self.get_vocal_presence:bool = getattr(self.h_params.process,"vocal_presence",False)
        self.apply_vocal_presence_to_spec:bool = getattr(self.h_params.process,"apply_vocal_presence_to_spec",False)
        
    def preprocess_input_for_inference(self,input,additional_dict=None) -> Tensor:
        spectrogram:Tensor = self.util.stft_torch(input)["mag"]
        mel_spec:Tensor = self.spec_to_mel(spectrogram)
        return mel_spec
    
    def postprocess_mel(self,mel) -> Tensor:
        return self.util.dynamic_range_compression(mel)

    def audio_to_hifi_gan_mel(self,audio) -> Tensor:
        spectrogram:Tensor = self.util.stft_torch(audio)["mag"]
        mel:Tensor =  self.util.spec_to_mel_spec(spectrogram)
        return self.util.dynamic_range_compression(mel)

    def spec_to_mel(self,spectrogram)->Tensor:
        mel_spec:Tensor = self.util.spec_to_mel_spec(spectrogram)
        return mel_spec
    
    def audio_to_spec_training_data(self,data_dict,additional_dict=None) -> dict:
        input_name:str = additional_dict["input_name"]
        output_name:str = additional_dict["target_name"]
        processed_feature_dict:dict[str,Tensor] = dict()
        processed_feature_dict[input_name] = self.util.stft_torch(data_dict[input_name])["mag"]
        processed_feature_dict[output_name] = self.util.stft_torch(data_dict[output_name])["mag"]

        if len(processed_feature_dict[input_name].shape) == 3:
            processed_feature_dict[input_name] = processed_feature_dict[input_name].unsqueeze(1)
            processed_feature_dict[output_name] = processed_feature_dict[output_name].unsqueeze(1)

        if self.h_params.pytorch_data.limiter_target_by_input:
            limiter_index = (processed_feature_dict[output_name] > processed_feature_dict[input_name])
            processed_feature_dict[output_name][limiter_index] = processed_feature_dict[input_name][limiter_index]
        
        if self.get_vocal_presence:
            processed_feature_dict["vocal_presence"] = self.util_vocal_presence.get_vocal_presence_from_raw_vocal_spec(processed_feature_dict[output_name])
            if self.apply_vocal_presence_to_spec:
                processed_feature_dict[output_name + "_vp_mask"] = self.util_vocal_presence.apply_vocal_presence_to_spec(processed_feature_dict[output_name],processed_feature_dict["vocal_presence"])

        return processed_feature_dict


    def preprocess_training_data(self,data_dict,additional_dict=None)->Dict[str,Tensor]:
        input_name:str = additional_dict["input_name"]
        output_name:str = additional_dict["target_name"]
        
        processed_feature_dict:dict[str,Tensor] = self.audio_to_spec_training_data(data_dict,additional_dict)
        
        processed_feature_dict[input_name] = self.spec_to_mel(processed_feature_dict[input_name])
        processed_feature_dict[output_name] = self.spec_to_mel(processed_feature_dict[output_name])
        if f"{output_name}_vp_mask" in processed_feature_dict:
            processed_feature_dict[f"{output_name}_vp_mask"] = self.spec_to_mel(processed_feature_dict[f"{output_name}_vp_mask"])
        return processed_feature_dict
    
