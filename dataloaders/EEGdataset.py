import torch
import scipy.io as scio
from scipy.signal import resample
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt, filtfilt, hilbert, iirnotch, lfilter, firwin
import librosa
import glob
import scipy
import os
import time
import random
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import EEG_ch_to_2D, region_split,get_k_list_with_minimal_hamming_distance, get_differential_entropy
from utils.cfg import base_path

    
class EEGDataset2(Dataset):  
    # dataset: dataset name. Das2016/Fug2018/Fug2020 
    # sub_id: list, which contains the index of the selected subjects
    # trial_list: 2-dimension list: (num_sub, num_trial)
    # time_index_list: 3-dimension list: (num_sub, num_trial, num_time_idx)
    # if_val: bool: whether divide the data into two part. One for validation and the other for testing
    # win_len: float: the length of the decision window (second)
    # band: list contains the type of the filter and the cutoff frequency. For example: ['bandpass', [1,31]]
    # topo: bool: whether to transform the channel dimension into 2D to align with the distribution of EEG scalp topography.
    # de: bool: whether compute the differential entropy of EEG signals
    def __init__(self, dataset, sub_id, trial_list, time_index_list, win_len, band=None, env_type='subband', topo=False, de=False, if_val=False):
        fs_eeg, fs_wav = 128, 8000
        if topo :
            assert de is False
        assert dataset == 'Das2016' or dataset == 'Fug2018' or dataset == 'Fug2020' 
        assert env_type == 'subband' or env_type == 'fullband' 
        assert (band is None) or (len(band) == 2)   
        self.de = de
        if_filt = False if band is None else True  
        if if_filt:
            if band[1][1] > 50:
                sos = [butter(5, [49.5, 50.5], 'bandstop', fs=128, output='sos'), butter(8, band[1], band[0], fs=128, output='sos')]
            else:
                sos = [butter(5, band[1], band[0], fs=128, output='sos')]
        self.eeg_all = []
        self.env_all = []
        self.wav_all = []
        self.target_direction_all = []
        self.target_gender_all = []
        # self.mode is useful only when if_val is True
        self.mode = 'test'
        self.if_val = if_val
        for idx, sub in enumerate(sub_id):
            with open(os.path.join(base_path, f'Data/{dataset}/preprocessed/S{sub}_preprocessed_py.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            for i in range(len(trial_list[idx])):
                trial_id = trial_list[idx][i]
                # 获取具体数据
                eeg = data['eeg'][trial_id]     # shape:(length, num_channel)
                if if_filt:
                    for filter in sos:
                        # print(filter)
                        eeg = sosfiltfilt(filter, eeg, axis=0) 
                eeg = (eeg - np.mean(eeg, 1, keepdims=True)).astype(np.float32)    
                ############## 变成二维拓扑版
                if topo:
                    eeg_2d = np.zeros((eeg.shape[0], 10, 11))
                    for i_ch in range(10):
                        for j_ch in range(11):
                            if EEG_ch_to_2D[i_ch][j_ch] > 0:
                                eeg_2d[:, i_ch, j_ch] = eeg[:, EEG_ch_to_2D[i_ch][j_ch] - 1]
                    eeg = eeg_2d.astype(np.float32)    # shape: (length, 10, 11)
                    del eeg_2d
                ##############
                env = data['envelope_subband'][trial_id].astype(np.float32) if env_type == 'subband' else data['envelope_fullband'][trial_id].astype(np.float32)
                wav = data['wav'][trial_id].astype(np.float32)
                target_gender = data['target_gender'][trial_id]
                target_direction = data['target_direction'][trial_id]

                eeg_this_trial = []
                env_this_trial = []
                wav_this_trial = []
                target_gender_this_trial = []
                target_direction_this_trial = []

                for t in time_index_list[idx][i]:

                    eeg_this_trial.append(eeg.copy()[int(t*fs_eeg): int(t*fs_eeg) + int(win_len * fs_eeg), ...])
                    # env_this_trial.append(env.copy()[int(t*fs_eeg): int(t*fs_eeg) + int(win_len * fs_eeg), :])
                    # wav_this_trial.append(wav.copy()[int(t*fs_wav): int(t*fs_wav) + int(win_len * fs_wav), :])
                    target_gender_this_trial.append(target_gender)
                    target_direction_this_trial.append(target_direction)
                
                if len(self.eeg_all) == 0:  # 列表内存占用较大，将数组以array形式储存
                    self.eeg_all = np.array(eeg_this_trial).astype(np.float32)
                    # self.env_all = np.array(env_this_trial).astype(np.float32)
                    # self.wav_all = np.array(wav_this_trial).astype(np.float32)
                else:
                    self.eeg_all = np.concatenate((self.eeg_all, np.array(eeg_this_trial).astype(np.float32)), 0).astype(np.float32)
                    # self.env_all = np.concatenate((self.env_all, np.array(env_this_trial).astype(np.float32)), 0).astype(np.float32)
                    # self.wav_all = np.concatenate((self.wav_all, np.array(wav_this_trial).astype(np.float32)), 0).astype(np.float32)

                
                
                self.target_direction_all = self.target_direction_all + target_direction_this_trial
                self.target_gender_all = self.target_gender_all + target_gender_this_trial
                del eeg, env, wav
            del data
        
        # not used in this experiment. Assign them to variables that occupies less memory
        self.env_all = self.target_direction_all 
        self.wav_all = self.target_direction_all 
        assert len(self.eeg_all) == len(self.env_all) == len(self.wav_all) == len(self.target_direction_all) == len(self.target_gender_all)
        
      

        self.len = len(self.eeg_all)

        
        self.eeg_all = np.array(self.eeg_all).astype(np.float32)
        self.env_all = np.array(self.env_all).astype(np.float32)
        self.wav_all = np.array(self.wav_all)
        self.target_direction_all = np.array(self.target_direction_all).astype(np.int32)
        self.target_gender_all = np.array(self.target_gender_all).astype(np.int32)
        


        
        index_all = list(range(self.len))
        random.shuffle(index_all)
        if self.if_val:
            index_list = index_all[0:self.len//2]
            index_list_val = index_all[self.len//2:]

            # self.eeg_all_val = self.eeg_all[index_list_val]
            self.eeg_all_val, self.env_all_val, self.wav_all_val, self.target_direction_all_val, self.target_gender_all_val = self.eeg_all[index_list_val], self.env_all[index_list_val], self.wav_all[index_list_val], self.target_direction_all[index_list_val], self.target_gender_all[index_list_val]

            self.eeg_all, self.env_all, self.wav_all, self.target_direction_all, self.target_gender_all = self.eeg_all[index_list], self.env_all[index_list], self.wav_all[index_list], self.target_direction_all[index_list], self.target_gender_all[index_list]

            
        

    def __getitem__(self, index):
        
        if self.if_val and self.mode == 'val':
            
            eeg = self.eeg_all_val[index]    
            if self.de:
                eeg = get_differential_entropy(eeg, fs=128)   # (C, 5)
            env = self.env_all_val[index]
            wav = self.wav_all_val[index]
            target_direction = self.target_direction_all_val[index]
            target_gender = self.target_gender_all_val[index]
        else:
            # print('bbb')
            eeg = self.eeg_all[index]
            if self.de:
                eeg = get_differential_entropy(eeg, fs=128)   # (C, 5)
            env = self.env_all[index]
            wav = self.wav_all[index]
            target_direction = self.target_direction_all[index]
            target_gender = self.target_gender_all[index]

        return eeg, env, wav, target_direction, target_gender
    
    def __len__(self):
        if self.if_val and self.mode == 'val':
            return len(self.eeg_all_val)
        else:

            return len(self.eeg_all)
    


    

class EEGDataset_wavelet(Dataset):   

    # dataset: dataset name. Das2016/Fug2018/Fug2020 
    # sub_id: list, which contains the index of the selected subjects
    # trial_list: 2-dimension list: (num_sub, num_trial)
    # time_index_list: 3-dimension list: (num_sub, num_trial, num_time_idx)
    # if_val: bool: whether divide the data into two part. One for validation and the other for testing
    # win_len: float: the length of the decision window (second)
    # band: list contains the type of the filter and the cutoff frequency. For example: ['bandpass', [1,31]]
    # prototype: int: the parameter K in the paper
    def __init__(self, dataset, sub_id, trial_list, time_index_list, win_len, band=None, env_type='subband', prototype=1, region_split=False, if_val=False):
        fs_eeg, fs_wav, fs_wavelet = 128, 8000, 10
        assert dataset == 'Das2016' or dataset == 'Fug2018' or dataset == 'Fug2020' 
        assert env_type == 'subband' or env_type == 'fullband' 
        assert (band is None) or (len(band) == 2)   # band 的格式应为 [bandtype, [f_1, f2]]
        assert not (if_val and prototype > 1)     # if_val=True只会用在测试模式上
        if_filt = False if band is None else True  # 根据band判断是否需要进行滤波
        if if_filt:
            if band[1][1] > 50:
                sos = [butter(5, [49.5, 50.5], 'bandstop', fs=128, output='sos'), butter(5, band[1], band[0], fs=128, output='sos')]
            else:
                sos = butter(5, band[1], band[0], fs=128, output='sos')
        self.eeg_all = []
        self.wavelet_all = []
        self.env_all = []
        self.wav_all = []
        self.target_direction_all = []
        self.target_gender_all = []
        self.prototype = prototype




        self.if_val = if_val
        self.mode = 'test'
        for idx, sub in enumerate(sub_id):
            with open(os.path.join(base_path, f'Data/{dataset}/preprocessed/S{sub}_preprocessed_py.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            for i in range(len(trial_list[idx])):
                trial_id = trial_list[idx][i]
                
                # 获取具体数据
                eeg = data['eeg'][trial_id]    # (T, channel)
                wavelet = data['wavelet_ref'][trial_id]   # (channel, T, F)
                if if_filt:
                    for filter in sos:
                        eeg = sosfiltfilt(filter, eeg, axis=0) 
                env = data['envelope_subband'][trial_id] if env_type == 'subband' else data['envelope_fullband'][trial_id]
                wav = data['wav'][trial_id]
                target_gender = data['target_gender'][trial_id]
                target_direction = data['target_direction'][trial_id]
                # 由于wavelet有nan，去除前0.5秒和最后0.5秒的数据
                eeg = eeg[int(fs_eeg*0.5):-int(fs_eeg*0.5), :].astype(np.float32)
                wavelet = wavelet[:, int(fs_wavelet*0.5):-int(fs_wavelet*0.5), :].astype(np.float32)
                wavelet = np.log(wavelet+1e-8)
                env = env[int(fs_eeg*0.5):-int(fs_eeg*0.5), :].astype(np.float32)
                wav = wav[int(fs_wav*0.5):-int(fs_wav*0.5), :].astype(np.float32)

                eeg_this_trial = []
                
                wavelet_this_trial = []
                env_this_trial = []
                wav_this_trial = []
                target_gender_this_trial = []
                target_direction_this_trial = []


                for t in time_index_list[idx][i]:
                    # eeg_this_trial.append(eeg.copy()[int(t*fs_eeg): int(t*fs_eeg) + int(win_len * fs_eeg), :])
                    wavelet_this_trial.append(wavelet.copy()[:, int(t*fs_wavelet): int(t*fs_wavelet) + int(win_len * fs_wavelet), :])
                    # env_this_trial.append(env.copy()[int(t*fs_eeg): int(t*fs_eeg) + int(win_len * fs_eeg), :])
                    target_gender_this_trial.append(target_gender)
                    target_direction_this_trial.append(target_direction)

                    



                # for each in env_this_trial:
                #     print(each.shape)
                if len(self.wavelet_all) == 0:  # 列表内存占用较大，将数组以array形式储存
                    # self.eeg_all = np.array(eeg_this_trial).astype(np.float32)
                    # self.env_all = np.array(env_this_trial).astype(np.float32)
                    
                    self.wavelet_all = np.array(wavelet_this_trial).astype(np.float32)
                else:
                    # self.eeg_all = np.concatenate((self.eeg_all, np.array(eeg_this_trial).astype(np.float32)), 0).astype(np.float32)
                    # self.env_all = np.concatenate((self.env_all, np.array(env_this_trial).astype(np.float32)), 0).astype(np.float32)
                    # self.wav_all = np.concatenate((self.wav_all, np.array(wav_this_trial).astype(np.float32)), 0).astype(np.float32)
                    self.wavelet_all = np.concatenate((self.wavelet_all, np.array(wavelet_this_trial).astype(np.float32)), 0).astype(np.float32)
                
                
                self.target_direction_all = self.target_direction_all + target_direction_this_trial
                self.target_gender_all = self.target_gender_all + target_gender_this_trial

                del eeg, wavelet, env, wav
            del data
        self.env_all = self.target_direction_all 
        self.wav_all = self.target_direction_all 
        self.eeg_all = self.target_direction_all 
        assert len(self.eeg_all) == len(self.wavelet_all) == len(self.env_all) == len(self.wav_all) == len(self.target_direction_all) == len(self.target_gender_all)


        self.len = len(self.wavelet_all)

        # 储存不同标签的index，便于后续一次性取出
        self.direction_0_index = []
        self.direction_1_index = []
        self.gender_0_index = []
        self.gender_1_index = []
        for k in range(self.len):
            if self.target_direction_all[k] == 0:
                self.direction_0_index.append(k)
            else:
                self.direction_1_index.append(k)

            if self.target_gender_all[k] == 0:
                self.gender_0_index.append(k)
            else:
                self.gender_1_index.append(k)


        self.wavelet_all = np.array(self.wavelet_all).astype(np.float32)
        self.env_all = np.array(self.env_all).astype(np.float32)
        self.wav_all = np.array(self.wav_all).astype(np.float32)
        self.target_direction_all = np.array(self.target_direction_all).astype(np.int32)
        self.target_gender_all = np.array(self.target_gender_all).astype(np.int32)

        index_all = list(range(self.len))
        random.shuffle(index_all)
        if self.if_val:
            index_list = index_all[0:self.len//2]
            index_list_val = index_all[self.len//2:]

            self.wavelet_all_val, self.env_all_val, self.wav_all_val, self.target_direction_all_val, self.target_gender_all_val = self.wavelet_all[index_list_val], self.env_all[index_list_val], self.wav_all[index_list_val], self.target_direction_all[index_list_val], self.target_gender_all[index_list_val]

            self.wavelet_all, self.env_all, self.wav_all, self.target_direction_all, self.target_gender_all = self.wavelet_all[index_list], self.env_all[index_list], self.wav_all[index_list], self.target_direction_all[index_list], self.target_gender_all[index_list]


        

    def __getitem__(self, index):
        
        
        if self.if_val and self.mode == 'val':
            

            env = self.env_all_val[index]
            wav = self.wav_all_val[index]
            target_direction = self.target_direction_all_val[index]
            target_gender = self.target_gender_all_val[index]
        else:
            # print('bbb')

            env = self.env_all[index]
            wav = self.wav_all[index]
            target_direction = self.target_direction_all[index]
            target_gender = self.target_gender_all[index]

        


        if self.prototype > 1:
            # prototype=random.randint(2, self.prototype)
            prototype=self.prototype
            if target_direction == 0:
                index_other = [self.direction_0_index[random.randint(0, len(self.direction_0_index) - 1)] for k in range(prototype - 1)] # 选出prototype-1个与原始数据标签相同的index
            else:
                index_other = [self.direction_1_index[random.randint(0, len(self.direction_1_index) - 1)] for k in range(prototype - 1)]

            
            wavelet_other = [self.wavelet_all[k] for k in index_other]
            wavelet_no_avg = [self.wavelet_all[index]] + wavelet_other
            wavelet_no_avg = np.array(wavelet_no_avg)
            wavelet_other.append(self.wavelet_all[index])    
            random_coeff = np.array([random.uniform(10.0, 100.0) for k in range(prototype)])
            random_coeff = random_coeff / np.sum(random_coeff)    # sum to 1
            for idx in range(len(wavelet_other)):
                wavelet_other[idx] = wavelet_other[idx] * random_coeff[idx]
            wavelet = np.sum(np.array(wavelet_other), 0)
            
        else:
            if self.if_val and self.mode == 'val':   # 如果是val模式，对wavelet重置
                wavelet = self.wavelet_all_val[index]
                wavelet_no_avg = np.expand_dims(wavelet, 0)
            else:
                wavelet = self.wavelet_all[index]
                wavelet_no_avg = np.expand_dims(self.wavelet_all[index], 0)
        



        




        return wavelet, env, wav, target_direction, target_gender, wavelet_no_avg
    
    def __len__(self):

        if self.if_val and self.mode == 'val':
            return len(self.wavelet_all_val)
        else:

            return len(self.wavelet_all)
        

   

