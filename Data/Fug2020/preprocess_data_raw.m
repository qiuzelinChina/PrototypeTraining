%This file  is modified from /src/examples/examplescript2.m in https://gitlab.com/sfugl/snhl

clear; clc; close all;


bidsdir =  pwd;
preprocessed_folder = 'preprocessed';
folderExists = exist(preprocessed_folder, 'dir');

if folderExists == 0
    mkdir(preprocessed_folder);
end



% We import information about the participants
participants    = readtable(fullfile(bidsdir,'participants.tsv'),'FileType','text','Delimiter','\t','TreatAsEmpty',{'N/A','n/a'});

dataout         = cell(44,1);
for subid = [21]
    Trials = {};
    Trials.eeg = {};
    Trials.envelope = {};
    Trials.wav = {};
    wav_att = {};
    wav_unatt = {};
    Trials.target_direction = [];
    Trials.target_gender = [];
    
    % The EEG data from each subject is stored in the following folder:
    eeg_dir     = fullfile(bidsdir,sprintf('sub-%0.3i',subid),'eeg');
    
    
    % The EEG data from sub-024 is split into two runs due to a break in the
    % experimental session. For this reason, we ensure that we loop over
    % these two runs for sub-024. For every other subject we do nothing.
    fname_bdf_file  = {};
    fname_events    = {};
    
    fname_bdf_file{1} = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_eeg.bdf',subid));
    fname_events{1} = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_events.tsv',subid));
    
    if subid == 24
        fname_bdf_file{2}     = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_run-2_eeg.bdf',subid));
        fname_events{2}       = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_run-2_events.tsv',subid));
    end
    
    
    
    
    
    % Prepare cell arrays that will contain EEG and audio features
    eegdat      = {};

    
    audiodat    = {};
    
    % Ensure that the script loops over both of the runs for sub-024
    for run = 1 : numel(fname_bdf_file)
        
        
        % Import the events that are stored in the .bdf EEG file. The
        % bdf_events table also contains information about which of the 
        % audio files that were presented during the EEG experiment. 
        bdf_events = readtable(fname_events{run},'FileType','text','Delimiter','\t','TreatAsEmpty',{'N/A','n/a'});
        
        % Select the rows in the table that points to onset triggers
        % (either onsets of target speech or onset of masker speech)
        bdf_target_masker_events = bdf_events(ismember(bdf_events.trigger_type,{'targetonset','maskeronset'}),:);
        
        
        fprintf('\n Importing data from sub-%0.3i',subid)
        fprintf('\n Preprocessing EEG data')
        
        
        % Preprocess the EEG data according to the proposed preprocessing
        % pipeline. Please inspect <preprocess_eeg> for more details. This
        % function can be found in the bottom of this script
        eegdat{run} = preprocess_eeg2(fname_bdf_file{run},bdf_events, 64, 0.1, 'no', 'yes');

        
       
        fprintf('\n Preprocessing audio data')
       
        
        
        index = 1;
        
        % we will store all of the wav-files in a cell of size (48 x 2) for
        % all subjects except sub-024 which has two runs
        wav_files = cell(sum(strcmp(bdf_target_masker_events.trigger_type,'targetonset')),2);

        for trial = 1 : size(bdf_target_masker_events,1)
            
            [fna,fnb,fnc] = fileparts(bdf_target_masker_events.stim_file{trial});
            
            
            % Import the audio files. Note that we here import audio
            % without CamEQ for the hearing-impaired listeners. 
            audio_type = 'woa'; 
            
            if strcmp(participants.hearing_status{subid},'nh')
                audio_type = '';
            end
            
            % Once this has been done, we can now import all of the audio
            % and store them in the <wav_files> cell.
            fname_audio = fullfile(bidsdir,'stimuli',[fna,'/',fnb,audio_type,fnc]);
            
            
            [wav,fsa] =  audioread(fname_audio);
            
            
            if strcmp(bdf_target_masker_events.trigger_type(trial),'maskeronset')
                
                % The row in <bdf_target_masker_events> points to a masker
                % wavfile. 
                
                % Figure out how delayed the masker onset was relative to 
                % the target speech onset. Once this has been done, use this
                % offset to pad the audio with silence. Note that we
                % originally did this in a different way but obtained
                % similar results!
                
                silence = bdf_target_masker_events(trial,:).onset - bdf_target_masker_events(trial-1,:).onset;
                
                % Zero-pad the masker audio with silence
                wav_files{index-1,2} = [zeros(round(silence*fsa),2); wav];
                
            else
                
                % The row in <bdf_target_masker_events> points to a target
                % wavfile. 
                
                if mod(index,5)==0 || index == 1
                    fprintf('\n Importing audio from trial %i',index)
                else
                    fprintf('.')
                end
                % Import the actual target speaker
                wav_files{index,1} = wav;
                
                index = index + 1;
            end
            
        end
        
               
        
        % We have now all of the wavfiles in one single cell. Extract audio
        % features from these wavefiles:
        audiofeat = {};
        audiofeat_lp32 = {};
        for trial = 1 : size(wav_files,1)
            
            % Extract the envelope feature for each audio stimuli
            [lp8, lp32] = preprocess_audio2(wav_files{trial,1},fsa);
            audiofeat{trial,1} =  lp8;
            audiofeat_lp32{trial,1} =  lp32;
            
            if mod(trial,5)==0 || trial == 1
                fprintf('\n Extracting audio feature from trial %i',trial)
            else
                fprintf('.')
            end
            
            if ~isempty(wav_files{trial,2})
                
                % Do exactly the same thing for envelopes of unattended
                % speech signals
                
                [lp8, lp32] = preprocess_audio2(wav_files{trial,2},fsa);
                audiofeat{trial,2} =  lp8;
                audiofeat_lp32{trial,2} =  lp32;
                
            end
            
            
        end
        
        
        % The subseqent audio representations will be stored in <audiodat> 
        % (taking into account the fact that sub-024 has data from two runs).
        audiodat = cat(1,audiodat,audiofeat);
        
    end
    
    

    % Append EEG data from the two runs for sub-024 
    if numel(eegdat)==2
        cfg = [];
        cfg.keepsampleinfo  = 'no';
        eegdat                 = ft_appenddata(cfg,eegdat{1},eegdat{2});
        
    else
        eegdat = eegdat{1};

    end
    
    
    
  
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Stimulus-response analysis
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    % Train stimulus-reconstruction model. Here, we focus on stimulus
    % reconstruction models with lambda parameters ranging between 10^(-3)
    % and 10^6. The models cover a range of lags ranging between 0 ms post
    % stimulus and 500 ms post stimulus
    
    tmin                    = 0;
    tmax                    = 500;
    fsr                     = 64;
    map                     = -1;
    lambda                  = logspace(-3,6,50);
    
    % Identify which trials that are considered to be single-talker
    
    single_talker_trials    = find(cellfun(@isempty,audiodat(:,2)));
    two_talker_trials       = find(~cellfun(@isempty,audiodat(:,2)));
    
    
    % Store the audio- and EEG- features in cells
    stim_att = cell(1,32);
    stim_itt = cell(1,32);
    resp_st  = cell(1,32);
    stim_st  = cell(1,16);
    resp_st  = cell(1,16);
    
    for ii = 1 : numel(two_talker_trials)
        
        % The envelopes of the attended and unattended speech stream are
        % stored in <stim_att> and <stim_itt> for the two-talker condition
        
        stim_att{1,ii}      = audiodat{two_talker_trials(ii),1};
        stim_itt{1,ii}      = audiodat{two_talker_trials(ii),2};
        
        % Similarly, for the EEG data we store data from the two-talker
        % condition in <resp_tt>
        
        resp_tt{1,ii}       = eegdat.trial{two_talker_trials(ii)}';
        
    end
    
    
    for ii = 1 : numel(single_talker_trials)
        
        % Store envelopes of attended speech in <stim_st> for the
        % single-talker condition
        
        stim_st{1,ii}     = audiodat{single_talker_trials(ii),1};
        
        % Store preprocessed EEG data (64 channels) <resp_st> for the
        % single-talker condition
        
        resp_st{1,ii}     = eegdat.trial{single_talker_trials(ii)}';
    end
    

   
    
    bdf_events_cp = table;
    size_bdf = size(bdf_events);
    j = 1;
    for i = 1: size_bdf(1)
        a = bdf_events.single_talker_two_talker(i, 1);

        if ~strcmp(a{1, 1}, 'n/a') && length(a{1,1})>0
            bdf_events_cp(j,:) = bdf_events(i, :);
            j = j + 1;
        end
    end
    num_trial = length(two_talker_trials);
    
    
    
    


    data = struct;

    data.eeg = cell(num_trial, 1);
    data.eeg_lp8 = cell(num_trial, 1);
    data.eeg_lp32 = cell(num_trial, 1);
    data.eeg_delta = cell(num_trial, 1);
    data.eeg_theta = cell(num_trial, 1);
    data.eeg_alpha = cell(num_trial, 1);
    data.eeg_beta = cell(num_trial, 1);
    data.eeg_gamma = cell(num_trial, 1);

    data.wav = cell(num_trial, 2);
    data.envelope_lp8 = cell(num_trial, 2);
    data.envelope_lp32 = cell(num_trial, 2);
    data.attend_mf = zeros(num_trial, 1);
    data.attend_lr = zeros(num_trial, 1);
    for i = 1:numel(two_talker_trials)
        attend_mf = bdf_events_cp.attend_male_female(two_talker_trials(i), 1);
        attend_lr = bdf_events_cp.attend_left_right(two_talker_trials(i), 1);
        if strcmp(attend_mf, 'attendmale')
            

            
            data.attend_mf(i) = 0;
            
            wav_0 = mean(resample(wav_files{two_talker_trials(i), 1}, 8000, 44100), 2);
            wav_1 = mean(resample(wav_files{two_talker_trials(i), 2}, 8000, 44100), 2);
            data.wav{i, 1} = wav_0(6*8000+1: 43*8000, :);
            data.wav{i, 2} = wav_1(6*8000+1: 43*8000, :);

            data.envelope_lp8{i, 1} = audiofeat{two_talker_trials(i), 1};
            data.envelope_lp8{i, 2} = audiofeat{two_talker_trials(i), 2};
            
            data.envelope_lp32{i, 1} = audiofeat_lp32{two_talker_trials(i), 1};
            data.envelope_lp32{i, 2} = audiofeat_lp32{two_talker_trials(i), 2};
            
            
            Trials.target_gender(i) = 1;
            
            
        else
            data.attend_mf(i) = 1;

            wav_0 = mean(resample(wav_files{two_talker_trials(i), 2}, 8000, 44100), 2);
            wav_1 = mean(resample(wav_files{two_talker_trials(i), 1}, 8000, 44100), 2);
            data.wav{i, 1} = wav_0(6*8000+1: 43*8000, :);
            data.wav{i, 2} = wav_1(6*8000+1: 43*8000, :);

            data.envelope_lp8{i, 1} = audiofeat{two_talker_trials(i), 2};
            data.envelope_lp8{i, 2} = audiofeat{two_talker_trials(i), 1};
            data.envelope_lp32{i, 1} = audiofeat_lp32{two_talker_trials(i), 2};
            data.envelope_lp32{i, 2} = audiofeat_lp32{two_talker_trials(i), 1};
            
            Trials.target_gender(i) = 0;
        end
        data.eeg{i, 1} = eegdat.trial{1, two_talker_trials(i)};
        
        if strcmp(attend_lr, 'attendleft')
            data.attend_lr(i) = 0;
            Trials.target_direction(i) = 0;
        else
            data.attend_lr(i) = 1;
            Trials.target_direction(i) = 1;
        end
        
        %%%%%%%%%%%%
        Trials.eeg{i} = eegdat.trial{1, two_talker_trials(i)}(:, 1:end-1).';
        Trials.wav{i} = cat(2, wav_0(6*8000+1: 43*8000, :), wav_1(6*8000+1: 43*8000, :));
        Trials.envelope{i} = cat(2, data.envelope_lp8{i, 1},data.envelope_lp8{i, 2});
        
        
        
    end
    data.two_talker_trials = two_talker_trials;
    
    %st
    num_trial_st = length(single_talker_trials);

    

    data.eeg_st = cell(num_trial_st, 1);
    

    data.wav_st = cell(num_trial_st, 1);
    data.envelope_st_lp8 = cell(num_trial_st, 1);
    data.envelope_st_lp32 = cell(num_trial_st, 1);
   
    for i = 1:numel(single_talker_trials)
        
     

        wav_0 = mean(resample(wav_files{single_talker_trials(i), 1}, 16000, 44100), 2);
       
        data.wav_st{i, 1} = wav_0(6*16000+1: 43*16000, :);
        

        data.envelope_st_lp8{i, 1} = audiofeat{single_talker_trials(i), 1};
        

        data.envelope_st_lp32{i, 1} = audiofeat_lp32{single_talker_trials(i), 1};
       
       
        data.eeg_st{i, 1} = eegdat.trial{1, single_talker_trials(i)};
        
    end
    data.single_talker_trials = single_talker_trials;
    
    
    
    
    
    
    name = [preprocessed_folder filesep 'S' num2str(subid), '_raw.mat'];
    save(name, 'Trials');
    
    
    
end
    
        
    









function dat = preprocess_eeg2(fname,bdf_events,lp, hp, if_lp, if_hp)

% Import the .bdf files
cfg=[];
cfg.channel = 'all';
cfg.dataset = fname;
dat = ft_preprocessing(cfg);


% Re-reference the EEG data
% cfg=[];
% cfg.reref       = 'yes';
% cfg.refchannel  = {'TP8','TP7'};
% dat = ft_preprocessing(cfg,dat);


% Define trials and segment EEG data using the events stored in the tsv
% files. Note that we here only focus on the target trials
% http://www.fieldtriptoolbox.org/example/making_your_own_trialfun_for_conditional_trial_definition/
        
bdf_target_events = bdf_events(strcmp(bdf_events.trigger_type,'targetonset'),:);



cfg             = [];
cfg.trl         = [ bdf_target_events.sample-5*dat.fsample, ...                     % start of segment (in samples re 0)
                    bdf_target_events.sample+50*dat.fsample, ...                    % end of segment
                    repmat(-5*dat.fsample,size(bdf_target_events.sample,1),1), ...  % how many samples prestimulus
                    bdf_target_events.value];                                       % store the trigger values in dat.trialinfo
dat             = ft_redefinetrial(cfg,dat);


if sum(sum(isnan(cat(1,dat.trial{:}))))
    error('Warning: For some reason there are nans produced. Please make sure that the trials are not defined to be too long')
end

cfg             = [];
cfg.resamplefs  = 128;
cfg.detrend     = 'no';
cfg.method      = 'resample';
dat             = ft_resampledata(cfg, dat);


% High-pass filter the EEG data
% cfg = [];
% cfg.hpfilter    = if_hp;
% cfg.hpfreq      = hp;
% cfg.hpfilttype  = 'but';
% dat             = ft_preprocessing(cfg,dat);


% Low-pass filter the EEG data
% cfg = [];
% cfg.lpfilter    = if_lp;
% cfg.lpfreq      = lp;
% cfg.lpfilttype  = 'but';
% dat             = ft_preprocessing(cfg,dat);

% Select a subset of electrodes
cfg = [];
cfg.channel     = 1:64;
dat             = ft_preprocessing(cfg,dat);


% We only focus on data from 6-s post stimulus onset to 43-s post
% stimulus onset
cfg             = [];
cfg.latency     = [6 43];
dat             = ft_selectdata(cfg, dat);

end






function feat = preprocess_audio(xx,fs)

% Define a minimalistic audio preprocessing pipeline. For simplicity, we
% here fullwave rectify the audio to obtain an estimate of the broadband
% envelope, then downsample the envelope representation to 64 Hz and 
% bandpass filter the envelope representation (using a 2nd order 
% acausal Butterworth filter). 

fsr = 128;
xx         = mean(xx,2);
xx         = resample(xx,12000,fs);

flow       = 100;
fhigh      = 4000;
fc         = erbspacebw(flow, fhigh);
[gb, ga]   = gammatone(fc, 12000, 'complex');

feat       = 2*real(ufilterbankz(gb,ga,xx));
feat       = abs(feat).^0.3;
feat       = mean(feat,2);
feat       = resample(feat,fsr,12000);

[bbp,abp] = butter(2,[1 9]/(fsr/2));
feat      = filtfilt(bbp,abp,feat);
feat      = feat(6*fsr+1:43*fsr,:);

end

function [feat_lp8, feat_lp32] = preprocess_audio2(xx,fs)

% Define a minimalistic audio preprocessing pipeline. For simplicity, we
% here fullwave rectify the audio to obtain an estimate of the broadband
% envelope, then downsample the envelope representation to 64 Hz and 
% bandpass filter the envelope representation (using a 2nd order 
% acausal Butterworth filter). 

fsr = 128;
xx         = mean(xx,2);
xx         = resample(xx,12000,fs);

flow       = 100;
fhigh      = 4000;
fc         = erbspacebw(flow, fhigh);
[gb, ga]   = gammatone(fc, 12000, 'complex');

feat       = 2*real(ufilterbankz(gb,ga,xx));
feat       = abs(feat).^0.3;
feat       = mean(feat,2);
feat       = resample(feat,fsr,12000);

[bbp1,abp1] = butter(2,[1 9]/(fsr/2));
feat_lp8      = filtfilt(bbp1,abp1,feat);
feat_lp8      = feat_lp8(6*fsr+1:43*fsr,:);

[bbp2,abp2] = butter(2,[1 32]/(fsr/2));
feat_lp32      = filtfilt(bbp2,abp2,feat);
feat_lp32     = feat_lp32(6*fsr+1:43*fsr,:);

end




