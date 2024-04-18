%filtering and wavelet transform
[b,a] = butter(5, [1 50]/(128/2));

for subid = 1:1
   
    
    
    
    load(['preprocessed' filesep 'S' num2str(subid) '_raw.mat']);
    Trials.wavelet = cell(1, length(Trials.eeg));
    Trials.wavelet_ref = cell(1, length(Trials.eeg));

    for i_trial = 1:length(Trials.eeg)
        data = {};
  
        data.label = cell(1,64);
        for i = 1:64
            data.label{i} = ['E' num2str(i)];
        end
        
        
        
        eeg_this_trial = Trials.eeg{i_trial};
        data.trial = filtfilt(b, a, eeg_this_trial).';
        data.time = (0:size(data.trial, 2)-1) / 128;
        data.fsample = 128;
        len = ceil(size(data.trial, 2) / 128);
        
        cfg = [];
        cfg.output = 'pow';
        cfg.channel = {'all'};
        cfg.pad = {'nextpow2'};
        cfg.method = 'wavelet';
        cfg.width = [linspace(1.5, 7, 20) 7*ones(1, 29)];
        cfg.output = 'pow';
        cfg.foi = 2:1:50;
        cfg.toi = 0:0.1:len-0.;
        cfg.keeptrials = 'yes';
        
        wavelet = ft_freqanalysis(cfg, data);
        spectrum =squeeze( wavelet.powspctrm(1, :, :, :));  % C, F, T
        Trials.wavelet{i_trial} = permute(spectrum,[1, 3, 2]); % C, F, T
        
        
        
        %%%%%%%%%%%ref
        data_ref = {};
  
        data_ref.label = cell(1,64);
        for i = 1:64
            data_ref.label{i} = ['E' num2str(i)];
        end
        
        
        
        eeg_this_trial = Trials.eeg{i_trial};
        eeg_this_trial = filtfilt(b, a, eeg_this_trial);
        data_ref.trial = (eeg_this_trial - mean(eeg_this_trial, 2)).';   % avg reference 

        data_ref.time = (0:size(data_ref.trial, 2)-1) / 128;
        data_ref.fsample = 128;
        len = ceil(size(data_ref.trial, 2) / 128);
        
        cfg = [];
        cfg.output = 'pow';
        cfg.channel = {'all'};
        cfg.pad = {'nextpow2'};
        cfg.method = 'wavelet';
        cfg.width = [linspace(1.5, 7, 20) 7*ones(1, 29)];
        cfg.output = 'pow';
        cfg.foi = 2:1:50;
        cfg.toi = 0:0.1:len-0.;
        cfg.keeptrials = 'yes';
        
        wavelet_ref = ft_freqanalysis(cfg, data_ref);
        spectrum_ref =squeeze( wavelet_ref.powspctrm(1, :, :, :));  % C, F, T
        Trials.wavelet_ref{i_trial} = permute(spectrum_ref,[1, 3, 2]); % C, F, T
    end
    
    for i_trial = 1:length(Trials.eeg)
        Trials.eeg{i_trial} = single(Trials.eeg{i_trial});
        Trials.envelope{i_trial} = single(Trials.envelope{i_trial});
        Trials.wav{i_trial} = single(Trials.wav{i_trial});
        Trials.wavelet{i_trial} = single(Trials.wavelet{i_trial});
        Trials.wavelet_ref{i_trial} = single(Trials.wavelet_ref{i_trial});
    end
    save(['preprocessed' filesep 'S' num2str(subid) '_preprocessed.mat'], 'Trials','-v6');
end

