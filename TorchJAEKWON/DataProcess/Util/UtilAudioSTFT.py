from numpy import ndarray

import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt

from HParams import HParams
from TorchJAEKWON.DataProcess.Util.UtilAudio import UtilAudio

class UtilAudioSTFT(UtilAudio):
    def __init__(self):
        super().__init__()
        self.h_params = HParams()

        self.nfft = self.h_params.preprocess.nfft
        self.hop_size = self.h_params.preprocess.hopsize
        self.hann_window = torch.hann_window(self.nfft)
    
    def get_mag_phase_stft_np(self,audio):
        stft = librosa.stft(audio,n_fft=self.h_params.preprocess.nfft, hop_length=self.h_params.preprocess.hopsize)
        mag = abs(stft)
        phase = np.exp(1.j * np.angle(stft))
        return {"mag":mag,"phase":phase}
    
    def get_mag_phase_stft_np_mono(self,audio):
        if audio.shape[0] == 2:
            return self.get_mag_phase_stft_np(np.mean(audio,axis=0))
        else:
            return self.get_mag_phase_stft_np(audio)

    
    def stft_torch(self,audio,eps: float = 0.0):
        audio_torch = audio
        if type(audio_torch) == np.ndarray:
            audio_torch = torch.from_numpy(audio_torch)
        
        if len(audio_torch.shape) >3:
            print(f"Error: stft_torch() audio torch shape is {audio_torch.shape}")
            exit()

        shape_is_three = True if len(audio_torch.shape) == 3 else False
        if shape_is_three:
            batch_size, channels_num, segment_samples = audio_torch.shape
            audio_torch = audio_torch.reshape(batch_size * channels_num, segment_samples)
            
        spec = torch.stft(  audio_torch,
                            n_fft=self.nfft,
                            hop_length=self.hop_size,
                            window=self.hann_window.to(audio_torch.device),
                            return_complex=True)
        mag = abs(spec)

        if shape_is_three:
            _, time_steps, freq_bins = mag.shape
            mag = mag.reshape(batch_size, channels_num, time_steps, freq_bins)

        return {"mag":mag}
    
    def get_pred_accom_by_subtract_pred_vocal_audio(self,pred_vocal,mix_audio):
        pred_vocal_mag = self.get_mag_phase_stft_np_mono(pred_vocal)["mag"]
        mix_stft = self.get_mag_phase_stft_np_mono(mix_audio)
        mix_mag = mix_stft["mag"]
        mix_phase = mix_stft["phase"]
        pred_accom_mag = mix_mag - pred_vocal_mag
        pred_accom_mag[pred_accom_mag < 0] = 0
        pred_accom = librosa.istft(pred_accom_mag*mix_phase,hop_length=self.h_params.preprocess.hopsize,length=mix_audio.shape[-1])
        return pred_accom
    
    def stft_plot_from_audio_path(self,audio_path:str,save_path:str = None, dpi:int = 500) -> None:
        audio, sr = librosa.load(audio_path)
        stft_audio:ndarray = librosa.stft(audio)
        spectrogram_db_scale:ndarray = librosa.amplitude_to_db(np.abs(stft_audio), ref=np.max)
        plt.figure(dpi=dpi)
        librosa.display.specshow(spectrogram_db_scale)
        plt.colorbar()
        if save_path is not None:
            plt.savefig(save_path,dpi=dpi)