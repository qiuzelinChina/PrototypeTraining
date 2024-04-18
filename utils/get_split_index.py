import numpy as np
import pickle
from sklearn.model_selection import KFold, LeaveOneOut
from utils.utils import divide_list
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cfg import base_path




def get_split_index_new(dataset_name, strategy, win_len, stride, sub_id_list, cv=4, target='direction', wavelet=False):
    """
    Obtain the start time of the decision window in a specific dataset under different conditions.
    strategy == 1:  Strategy I in the paper
    strategy == 2:  Strategy II in the paper
    strategy == 3:  Strategy III in the paper
    """
    # 判断目标是方位还是性别
    assert target == 'direction' or target == 'gender'
    if strategy == 1 or strategy == 2 or strategy == 3:     

        assert len(sub_id_list) == 1           
    kf = KFold(n_splits=cv, shuffle=True)


    for sub_idx, sub_id in enumerate(sub_id_list):


        if strategy == 1:       
            sub_id = sub_id_list[0]  
            with open(os.path.join(base_path, f'Data/{dataset_name}/preprocessed/S{sub_id}_preprocessed_py.pkl'), 'rb') as f:
                data = pickle.load(f)
            trial_list_train = [[] for k in range(cv)]
            trial_list_test = [[] for k in range(cv)]
            sub_list_train = [[sub_id] for k in range(cv)]
            sub_list_test = [[sub_id] for k in range(cv)]
            time_index_list_train = []
            time_index_list_test = []
            num_trials = len(data['eeg'])
            all_trials = list(range(num_trials))

        
            if dataset_name == 'Das2016':
                all_trials_2 = all_trials[8:20]    # 6 min trials
                all_trials = all_trials[0:8]      # 2 min trials   


            # 为了保证训练集上每种标签的数据尽可能均匀，将两种标签不同的数据分开，而后分别进行split
            # To ensure that each label’s data in the training set is as evenly distributed as possible, 
            # separate the data with different labels, and then perform the split separately.
            target_0_trial = []
            target_1_trial = []
            target_0_trial_2 = []   # for 2-min trials
            target_1_trial_2 = []
            target_label = data['target_direction'] if target == 'direction' else data['target_gender']
            for idx_trial in all_trials:
                if target_label[idx_trial] == 0:
                    target_0_trial.append(idx_trial)
                else:
                    target_1_trial.append(idx_trial)

            if dataset_name == 'Das2016':
                for idx_trial in all_trials_2:
                    if target_label[idx_trial] == 0:
                        target_0_trial_2.append(idx_trial)
                    else:
                        target_1_trial_2.append(idx_trial) 
                
            # 对target为0和为1的数据进行两次N-fold划分 
            # Perform N-fold partitioning separately for data with target values of 0 and 1
            for i, (train, test) in enumerate(kf.split(target_0_trial)):
                trial_list_train[i] = trial_list_train[i] + [target_0_trial[each] for each in list(train)]
                trial_list_test[i] = trial_list_test[i] + [target_0_trial[each] for each in list(test)]
            for i, (train, test) in enumerate(kf.split(target_1_trial)):
                trial_list_train[i] = trial_list_train[i] + [target_1_trial[each] for each in list(train)]
                trial_list_test[i] = trial_list_test[i] + [target_1_trial[each] for each in list(test)]
            
            # 如果数据集是Das2016，则将第二部分数据也划分出来
            # if the dataset is Das2016，do the same thing for the 2-min trials
            if dataset_name == 'Das2016':
                for i, (train, test) in enumerate(kf.split(target_0_trial_2)):
                    trial_list_train[i] = trial_list_train[i] + [target_0_trial_2[each] for each in list(train)]
                    trial_list_test[i] = trial_list_test[i] + [target_0_trial_2[each] for each in list(test)]
                for i, (train, test) in enumerate(kf.split(target_1_trial_2)):
                    trial_list_train[i] = trial_list_train[i] + [target_1_trial_2[each] for each in list(train)]
                    trial_list_test[i] = trial_list_test[i] + [target_1_trial_2[each] for each in list(test)]

            for i in range(cv):
                trial_list_train[i] = [trial_list_train[i]]   # add the dimension of subject. shape: [cv, 1, num_trial]
                trial_list_test[i] = [trial_list_test[i]]
            for i in range(len(trial_list_train)): # shape (num_fold , num_trial , num_time_index_each_trial)
                index_train_this_fold = []
                index_test_this_fold = []
                for j in range(len(trial_list_train[i][0])):
                    len_eeg = int(data['eeg'][trial_list_train[i][0][j]].shape[0] / 128)
                    len_eeg = len_eeg- 1 if wavelet else len_eeg
                    time_index_list = list(np.arange(0, len_eeg-win_len, stride))   
                    index_train_this_fold.append(time_index_list)

                for j in range(len(trial_list_test[i][0])):
                    len_eeg = int(data['eeg'][trial_list_test[i][0][j]].shape[0] / 128)
                    len_eeg = len_eeg- 1 if wavelet else len_eeg
                    time_index_list = list(np.arange(0, len_eeg-win_len, stride))   
                    index_test_this_fold.append(time_index_list)
                time_index_list_train.append([index_train_this_fold])   # shape: [cv, 1, num_trial, num_time_index]
                time_index_list_test.append([index_test_this_fold])

            



        # 先将每个窗口都划分为决策窗，而后再进行shuffle和partition
        # split each trial into decision windows and then shuffle and partition
        if strategy == 3:
            sub_id = sub_id_list[0]  
            with open(os.path.join(base_path, f'Data/{dataset_name}/preprocessed/S{sub_id}_preprocessed_py.pkl'), 'rb') as f:
                data = pickle.load(f)
            trial_list_train = [[] for k in range(cv)]
            trial_list_test = [[] for k in range(cv)]
            sub_list_train = [[sub_id] for k in range(cv)]
            sub_list_test = [[sub_id] for k in range(cv)]
            time_index_list_train = []
            time_index_list_test = []
            num_trials = len(data['eeg'])
            all_trials = list(range(num_trials))
            all_trial_time_index = []    # save all [trial, time_index, label] pairs
            for trial_idx in all_trials:
                len_eeg = int(data['eeg'][trial_idx].shape[0] / 128)
                len_eeg = len_eeg- 1 if wavelet else len_eeg
                time_index_list = list(np.arange(0, len_eeg-win_len, win_len))     # 在此种策略下，窗长应该严格等于帧移  under this stragegy, window length equals stride strictly
                target_label = data['target_direction'][trial_idx] if target == 'direction' else data['gender'][trial_idx]   
                for each in time_index_list:
                    all_trial_time_index.append([trial_idx, each, target_label])
            for i, (train, test) in enumerate(kf.split(all_trial_time_index)):
                this_fold_train_dict, this_fold_test_dict = {}, {}
                index_train_this_fold = []
                index_test_this_fold = []

                # 针对train
                for each in train:
                    this_fold_train_dict[f'{all_trial_time_index[each][0]}'] = []        
                for each in train:
                    this_fold_train_dict[f'{all_trial_time_index[each][0]}'].append(all_trial_time_index[each][1])
                for each in this_fold_train_dict.keys():
                    trial_list_train[i].append(int(each))
                    index_train_this_fold.append(this_fold_train_dict[each])
                time_index_list_train.append(index_train_this_fold)

                for each in test:
                    this_fold_test_dict[f'{all_trial_time_index[each][0]}'] = []       
                for each in test:
                    this_fold_test_dict[f'{all_trial_time_index[each][0]}'].append(all_trial_time_index[each][1])
                for each in this_fold_test_dict.keys():
                    trial_list_test[i].append(int(each) )
                    index_test_this_fold.append(this_fold_test_dict[each])
                time_index_list_test.append(index_test_this_fold)

            for i in range(cv):
                trial_list_train[i] = [trial_list_train[i]]   # add the dimension of subject. shape: [cv, 1, num_trial]
                trial_list_test[i] = [trial_list_test[i]]
                time_index_list_train[i] = [time_index_list_train[i]]
                time_index_list_test[i] = [time_index_list_test[i]]
        # 将每个trial划分为CV个整段，然后从整段中选取
        if strategy == 2:
            sub_id = sub_id_list[0] 
            with open(os.path.join(base_path, f'Data/{dataset_name}/preprocessed/S{sub_id}_preprocessed_py.pkl'), 'rb') as f:
                data = pickle.load(f)
            trial_list_train = [[] for k in range(cv)]
            trial_list_test = [[] for k in range(cv)]
            sub_list_train = [[sub_id] for k in range(cv)]
            sub_list_test = [[sub_id] for k in range(cv)]
            time_index_list_train = [[] for k in range(cv)]
            time_index_list_test = [[] for k in range(cv)]
            num_trials = len(data['eeg'])
            all_trials = list(range(num_trials))
            all_trial_time_index = []    # save all [trial, time_index, label] pairs
            for trial_idx in all_trials:
                len_eeg = int(data['eeg'][trial_idx].shape[0] / 128)
                len_eeg = len_eeg- 1 if wavelet else len_eeg
                time_index_list = list(np.arange(0, len_eeg-win_len, stride))     
                # print(time_index_list)
                time_index_list = divide_list(time_index_list, cv)     # 将time_index列表尽可能可能分为CV段 Divide the time_index list into CV segments as evenly as possible
                target_label = data['target_direction'][trial_idx] if target == 'direction' else data['gender'][trial_idx]   
                for i, (train, test) in enumerate(kf.split(time_index_list)):

                    trial_list_test[i].append(trial_idx)
                    tmp_test = []
                    for each in test:
                        tmp_test = tmp_test + time_index_list[each]

                    time_index_list_test[i].append(tmp_test.copy())

                    

                    trial_list_train[i].append(trial_idx)
                    tmp = []
                    tmp_train = []
                    for each in train:
                        tmp = tmp + time_index_list[each]
                    # 需要排除train中与test有交叠的部分 delete the decision windows that overlap with the testset
                    for t_idx_train in tmp:
                        if_keep = True
                        for t_idx_test in tmp_test:
                            if (t_idx_test + stride > t_idx_train and t_idx_test + stride < t_idx_train + stride) or (t_idx_test > t_idx_train and t_idx_test < t_idx_train + stride):
                                if_keep = False
                        if if_keep:
                            tmp_train.append(t_idx_train)
                    
                    time_index_list_train[i].append(tmp_train.copy())
                    del tmp, tmp_train, tmp_test


            for i in range(cv):
                trial_list_train[i] = [trial_list_train[i]]    # add the dimension of subject. shape: [cv, 1, num_trial]
                trial_list_test[i] = [trial_list_test[i]]
                time_index_list_train[i] = [time_index_list_train[i]]
                time_index_list_test[i] = [time_index_list_test[i]]  



                    

    return trial_list_train, trial_list_test, time_index_list_train, time_index_list_test, sub_list_train, sub_list_test 



if __name__ == '__main__':
    a  = 1