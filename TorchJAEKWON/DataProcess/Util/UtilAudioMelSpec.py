import torch
import numpy as np
from HParams import HParams
from DataProcess.Util.UtilAudioSTFT import UtilAudioSTFT

from librosa.filters import mel as librosa_mel_fn

class UtilAudioMelSpec(UtilAudioSTFT):
    def __init__(self, h_params: HParams):
        super().__init__(h_params)

        self.sample_rate = self.h_params.preprocess.sample_rate
        self.mel_size = self.h_params.preprocess.mel_size
        self.frequency_min = self.h_params.preprocess.fmin
        self.frequency_max = self.h_params.preprocess.fmax

        #self.mel_basis.shape == (self.mel_size, self.nfft//2 + 1)
        self.mel_basis = librosa_mel_fn(self.sample_rate, self.nfft, self.mel_size, self.frequency_min, self.frequency_max)
    
    def spec_to_mel_spec(self,stft_mag):
        if type(stft_mag) == np.ndarray:
            return np.matmul(self.mel_basis, stft_mag)
        elif type(stft_mag) == torch.Tensor:
            torch_mel_basis = torch.from_numpy(self.mel_basis).float().to(stft_mag.device)
            return torch.matmul(torch_mel_basis, stft_mag)
        else:
            print("spec_to_mel_spec type error")
            exit()
    
    def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
        if type(x) == np.ndarray:
            return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
        elif type(x) == torch.Tensor:
            return torch.log(torch.clamp(x, min=clip_val) * C)
        else:
            print("dynamic_range_compression type error")
            exit()