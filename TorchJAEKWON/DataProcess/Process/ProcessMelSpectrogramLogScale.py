import torch
import numpy as np

from HParams import HParams

from DataProcess.Process.ProcessMelSpectrogram import ProcessMelSpectrogram
from DataProcess.Util.UtilAudioMelSpec import UtilAudioMelSpec


class ProcessMelSpectrogramLogScale(ProcessMelSpectrogram):
    def __init__(self,h_params:HParams):
        super(ProcessMelSpectrogramLogScale,self).__init__(h_params)
        self.min_value_log_scale_mel = torch.log(torch.Tensor([1e-5]))
        
    def preprocess_input_for_inference(self,input,additional_dict=None):
        spectrogram = self.util.stft_torch(input)["mag"]
        mel_spec = self.util.spec_to_mel_spec(spectrogram)
        log_scale_mel = self.util.dynamic_range_compression(mel_spec)
        positive_mel = log_scale_mel - self.min_value_log_scale_mel.to(spectrogram.device)
        return positive_mel
    
    def model_output_to_hifi_gan_mel(self,mel):
        log_scale_mel = mel + self.min_value_log_scale_mel.to(mel.device)
        return log_scale_mel

    def spec_to_mel(self,spectrogram):
        mel_spec = self.util.spec_to_mel_spec(spectrogram)
        log_scale_mel = self.util.dynamic_range_compression(mel_spec)
        positive_mel = log_scale_mel - self.min_value_log_scale_mel.to(spectrogram.device)
        return positive_mel