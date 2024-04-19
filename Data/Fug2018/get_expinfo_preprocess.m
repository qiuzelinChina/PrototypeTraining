% remove the trial info of the single-speaker trial
for sub = 1:4
    expinfo_prepro = {};
    expinfo_prepro.attend_mf = cell(0,0); 
    expinfo_prepro.attend_lr = cell(0,0); 
    expinfo_prepro.acoustic_condition = cell(0,0); 
    expinfo_prepro.n_speakers = cell(0,0);
    expinfo_prepro.wavefile_male = cell(0,0); 
    expinfo_prepro.wavefile_female = cell(0,0); 
    expinfo_prepro.trigger = cell(0,0);

    load(['EEG' filesep 'S' num2str(sub) '.mat']);


    for i = 1:size(expinfo, 1)
        if expinfo{i, 4} == 2
            expinfo_prepro.attend_mf = [expinfo_prepro.attend_mf; expinfo{i, 1}];
            expinfo_prepro.attend_lr = [expinfo_prepro.attend_lr; expinfo{i, 2}];
            expinfo_prepro.acoustic_condition = [expinfo_prepro.acoustic_condition; expinfo{i, 3}];
            expinfo_prepro.n_speakers = [expinfo_prepro.n_speakers; expinfo{i, 4}];
            expinfo_prepro.wavefile_male = [expinfo_prepro.wavefile_male; expinfo{i, 5}];
            expinfo_prepro.wavefile_female = [expinfo_prepro.wavefile_female; expinfo{i, 6}];
            expinfo_prepro.trigger = [expinfo_prepro.trigger; expinfo{i, 7}];
        end
    end
    assert(length(expinfo_prepro.attend_mf)==60 && length(expinfo_prepro.attend_lr)==60 && length(expinfo_prepro.acoustic_condition)==60 ...
       && length(expinfo_prepro.n_speakers)==60 && length(expinfo_prepro.wavefile_male)==60&& length(expinfo_prepro.wavefile_female)==60 ...
       && length(expinfo_prepro.trigger)==60);
    save(['DATA_preproc' filesep  'expinfo_prepro_S' num2str(sub) '.mat'], 'expinfo_prepro');
end
