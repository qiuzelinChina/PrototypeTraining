import math
import sys
import csv
import sys
import os
import itertools
from scipy.signal import welch
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def view_bar(message, num, total, batch_mean_loss, all_mean_loss):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s: batch loss: %.05f   \t all loss: %.05f   \t [%s%s]  %d %%  \t%d/%d' % (message, batch_mean_loss, all_mean_loss, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    #r = '\r%s: snr: %.05f   \t pesq: %.05f   \t loss: %.05f   \t mean snr: %.05f   \t mean pesq: %.05f   \t mean loss: %.05f   \t [%s%s]  %d %%  \t%d/%d' % (message, snr_loss, pesq_loss, loss, mean_snr, mean_pesq, mean_loss, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

def write_log(log_path, content,mode='a'):
    with open(log_path, mode) as f:
        writer = csv.writer(f) 
        writer.writerow(content)


EEG_ch_to_2D = [[-1 ,-1 ,-1 ,-1 , 1 ,33 ,34 ,-1, -1 ,-1 ,-1],
[-1 ,-1, -1 , 2  ,3, 37 ,36, 35, -1 ,-1 ,-1],
[-1, 7 , 6 , 5 , 4 ,38 ,39 ,40 ,41 ,42 ,-1],
[-1 , 8 , 9 ,10 ,11 ,47 ,46 ,45 ,44 ,43 ,-1],
[-1 ,15, 14 ,13 ,12 ,48 ,49 ,50 ,51 ,52, -1],
[-1, 16 ,17 ,18, 19, 32 ,56 ,55 ,54, 53, -1],
[24 ,23 ,22 ,21 ,20, 31 ,57 ,58 ,59 ,60 ,61],
[-1, -1 ,-1, 25, 26 ,30 ,63 ,62 ,-1, -1, -1],
[-1, -1 ,-1, -1 ,27 ,29 ,64, -1, -1 ,-1 ,-1],
[-1 ,-1 ,-1 ,-1 ,-1 ,28 ,-1 ,-1,-1 ,-1 ,-1]]




######## 与自监督学习有关#############
# 将电极划分为8个区域，每个区域内有8个电极
region_split = [[1, 33, 34, 2, 3, 37, 36, 55], 
                [7, 6, 5, 4, 8, 9, 10, 11],
                [42, 41, 40, 39, 43, 44, 45, 46],
                [15, 14, 13, 12, 16, 17, 18, 19],
                [52, 51, 50, 49, 53, 54, 55, 56],
                [24, 23, 22, 21, 20, 25, 26, 27],
                [61, 60, 59, 58, 57, 62, 63, 64],
                [38, 47, 48, 32, 31, 30, 29, 28]]
original_list = [0, 1, 2, 3, 4, 5, 6, 7]
def hamming_distance(list1, list2):
    # 初始化距离为0
    distance = 0
    # 遍历两个列表的对应位置
    for i in range(len(list1)):
        # 如果元素不同，距离加1
        if list1[i] != list2[i]:
            distance += 1
    # 返回距离
    return distance

def get_k_list_with_minimal_hamming_distance(original_list=[0, 1, 2, 3, 4, 5, 6, 7], k=128):

    # 生成原列表的所有可能的排列
    permutations = list(itertools.permutations(original_list))

    # 根据hamming距离的大小，对所有的排列进行排序
    sorted_permutations = sorted(permutations, key=lambda x: hamming_distance(x, original_list))

    # 取出排序后的前128个排列，即为与原列表hamming距离最小的128种组合
    min_hamming_permutations = sorted_permutations[:k]
    min_hamming_permutations = [list(each) for each in min_hamming_permutations]
    # 打印结果
    return min_hamming_permutations


min_hamming_permutations = get_k_list_with_minimal_hamming_distance()
min_hamming_permutations_256 = get_k_list_with_minimal_hamming_distance(k=256)

###########################
    

# 计算各频带微分熵
def get_differential_entropy(eeg_signal, fs=128):
    # 定义频带范围
    bands = {
        'delta': (1, 3),
        'theta': (4, 7),
        'alpha': (8, 13),
        'beta': (14, 30),
        'gamma': (31, 50)
    }
  
    # 初始化微分熵字典
    de = {band: None for band in bands}
    
    # 对每个频带计算微分熵
    for band in bands:
        # 使用Welch方法计算功率谱密度
        f, Pxx = welch(eeg_signal.T, fs=fs, nperseg=fs, axis=-1)
        
        # 找到当前频带的索引
        idx_band = np.logical_and(f >= bands[band][0], f <= bands[band][1])
        
        # 计算当前频带的功率谱密度
        psd_band = Pxx[:, idx_band]
        
        # 计算微分熵
        de[band] = -np.sum(psd_band * np.log(psd_band), axis=-1)
    
    de_list = [de[band] for band in de.keys()]
    de_list = np.stack(de_list, -1)
    return de_list

def get_correct_num_ptl(out_emb, target_direction, pt0, pt1, mode = 1):
    # print(pt0)
    # print(pt1)
    num_correct = 0
    num_false = 0
    d0 =  torch.sum((out_emb.unsqueeze(-1) - pt0.unsqueeze(0)) ** 2, 1)   # (B, 10)
    d1 =  torch.sum((out_emb.unsqueeze(-1) - pt1.unsqueeze(0)) ** 2, 1)   # (B, 10)

    d01 = torch.cat((d0, d1), 1)
    d01_sort, sort_idx = torch.sort(d01, 1)
    label_concat = torch.cat((torch.zeros((d0.shape[0], d0.shape[1])), torch.ones((d1.shape[0], d1.shape[1]))), 1)
    # label01 = torch.zeros((d0.shape[1]*2))

    d0_with_label = torch.cat((d0.unsqueeze(-1), torch.zeros((d0.shape[0], d0.shape[1], 1)).to(d0.device)), -1)   # (B, 10, 2)
    d1_with_label = torch.cat((d1.unsqueeze(-1), torch.ones((d1.shape[0], d1.shape[1], 1)).to(d1.device)), -1)   # (B, 10, 2)
    d01_with_label, _ = torch.sort(torch.cat((d0_with_label, d1_with_label), 1), 1)     # # (B, 20, 2)

    # print(d01_with_label[b,:,1])
    d0_min, _ = torch.min(d0, dim=1)
    d1_min, _ = torch.min(d1, dim=1)
    for b in range(out_emb.shape[0]):
        if mode == 1:
            if (d0_min[b] < d1_min[b] and target_direction[b] == 0) or (d0_min[b] > d1_min[b] and target_direction[b] == 1):
                num_correct = num_correct + 1
        elif mode == 2:
            label01 = torch.stack([label_concat[b][each] for each in sort_idx[b][0:7]])

            # print(torch.mean(label01))
            if (torch.mean(label01) < 0.5 and target_direction[b] == 0) or (torch.mean(label01) > 0.5 and target_direction[b] == 1):
                num_correct = num_correct + 1
    
    return num_correct


def get_final_accuracy(accuracy_list):

    accuracy_list = torch.tensor(np.array(accuracy_list))


    _, test_max_index = torch.max(accuracy_list[:, 0], 0)

    _, val_max_index = torch.max(accuracy_list[:, 1], 0)

    acc = float((accuracy_list[test_max_index, 1] + accuracy_list[val_max_index, 0]) / 2)

    return acc


def divide_list(lst, n):
    """将列表lst尽可能平均分成n段"""
    k, m = divmod(len(lst), n)
    return list(lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))



