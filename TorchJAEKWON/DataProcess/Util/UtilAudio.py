import numpy as np
import soundfile as sf
import librosa

from HParams import HParams

class UtilAudio:
    def float32_to_int16(self, x: np.float32) -> np.int16:
        x = np.clip(x, a_min=-1, a_max=1)
        return (x * 32767.0).astype(np.int16)
    
    def int16_to_float32(self, x: np.int16) -> np.float32:
        return (x / 32767.0).astype(np.float32)
    
    def resample_audio(self,audio,origin_sr,target_sr,resample_type = "kaiser_fast"):
        print(f"resample audio {origin_sr} to {target_sr}")
        return librosa.core.resample(audio, orig_sr=origin_sr, target_sr=target_sr, res_type=resample_type)
    
    def read_audio_fix_channels_of_mono_stereo(self,audio_path,sample_rate=None, mono=False,read_type="librosa"):
        if read_type == "soundfile":
            audio_data, original_samplerate = sf.read(audio_path)

            if sample_rate is not None:
                print(f"resample audio {original_samplerate} to {sample_rate}")
                audio_data = self.resample_audio(audio_data,original_samplerate,sample_rate)

            if mono and audio_data.shape[1] == 2:
                audio_data = np.mean(audio_data,axis=1)
            elif not mono and len(audio_data.shape) == 1:
                stereo_audio = np.zeros((len(audio_data),2))
                stereo_audio[...,0] = audio_data
                stereo_audio[...,1] = audio_data
                audio_data = stereo_audio
        elif read_type == "librosa":
            print(f"read audio sr: {sample_rate}")
            audio_data, _ = librosa.core.load( audio_path, sr=sample_rate, mono=mono)
            
        return audio_data
