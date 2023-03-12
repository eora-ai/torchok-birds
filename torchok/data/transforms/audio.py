import numpy as np
import cv2
import librosa
import glob
from albumentations import BasicTransform, DualTransform

from torchok.constructor import TRANSFORMS


@TRANSFORMS.register_class
class AudioTransform(BasicTransform):
    """ Transform for audio task. This is the main class where we override the targets and update params function for our need"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


@TRANSFORMS.register_class
class AudioMinMaxNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, data, **params):
        max_vol = np.abs(data).max()
        y_vol = data * 1 / max_vol
        return np.asfortranarray(y_vol)


@TRANSFORMS.register_class
class AudioTimeShifting(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self ,data, **params):
        '''
        data : ndarray of audio timeseries
        '''        
        start_ = int(np.random.uniform(-80000, 80000))
        if start_ >= 0:
            audio_time_shift = np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)]
        else:
            audio_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]]
        
        return audio_time_shift


@TRANSFORMS.register_class
class AudioSpeedTuning(AudioTransform):
    """ Do speed Tuning of audio """
    def __init__(self, always_apply=False, p=0.5,speed_rate = None):
        '''
        Give Rate between (0.5,1.5) for best results
        '''
        super().__init__(always_apply, p)
        
        if speed_rate:
            self.speed_rate = speed_rate
        else:
            self.speed_rate = np.random.uniform(0.6, 1.3)
        
    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''        
        audio_speed_tune = cv2.resize(data, (1, int(len(data) * self.speed_rate))).squeeze()
        if len(audio_speed_tune) < len(data) :
            pad_len = len(data) - len(audio_speed_tune)
            audio_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                   audio_speed_tune,
                                   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(audio_speed_tune) - len(data)
            audio_speed_tune = audio_speed_tune[int(cut_len/2):int(cut_len/2)+len(data)]
        
        return audio_speed_tune


@TRANSFORMS.register_class
class AudioStretchAudio(AudioTransform):
    """ Do stretching of audio file"""
    def __init__(self, always_apply=False, p=0.5 , rate = None):
        super().__init__(always_apply, p)
        
        if rate:
            self.rate = rate
        else:
            self.rate = np.random.uniform(0.5, 1.5)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        input_length = len(data)
        
        data = librosa.effects.time_stretch(data, self.rate)
        
        if len(data)>input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data


@TRANSFORMS.register_class
class AudioPitchShift(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5, n_steps=None):
        super().__init__(always_apply, p)
        '''
        nsteps here is equal to number of semitones
        '''
        
        self.n_steps = n_steps
        
    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''        
        return librosa.effects.pitch_shift(data, sr=22050, n_steps=self.n_steps)


@TRANSFORMS.register_class
class AudioAddGaussianNoise(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
        
    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        ''' 
        noise = np.random.randn(len(data))
        data_wn = data + 0.005 * noise
        return data_wn


@TRANSFORMS.register_class
class AudioAddCustomNoise(AudioTransform):
    """
    This Function allows you to add noise from any custom file you want just give path to the directory where the files
    are stored and you are good to go.
    """
    def __init__(self,file_dir, always_apply=False, p=0.5 ):
        super().__init__(always_apply, p)
        '''
        file_dir must be of form '.../input/.../something'
        '''
        
        self.noise_files = glob.glob(file_dir+'/*')
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        ''' 
        nf = self.noise_files[int(np.random.uniform(0, len(self.noise_files)))]
        
        noise,_ = librosa.load(nf)
        
        if len(noise)>len(data):
            start_ = np.random.randint(len(noise) - len(data))
            noise = noise[start_: start_ + len(data)] 
        else:
            noise = np.pad(noise, (0, len(data)-len(noise)), "constant")
            
        data_wn= data  + noise

        return data_wn


@TRANSFORMS.register_class
class AudioPolarityInversion(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''
        return -data


@TRANSFORMS.register_class
class AudioGain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False,p=0.5):
        super().__init__(always_apply,p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db


    def apply(self, data, **args):
        amplitude_ratio = 10**(np.random.uniform(self.min_gain_in_db, self.max_gain_in_db)/20)
        return data * amplitude_ratio


@TRANSFORMS.register_class
class AudioCutOut(AudioTransform):
    def __init__(self, always_apply=False, p=0.5 ):
        super().__init__(always_apply, p)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''
        start_ = np.random.randint(0,len(data))
        end_ = np.random.randint(start_,len(data))
        
        data[start_:end_] = 0
        
        return data