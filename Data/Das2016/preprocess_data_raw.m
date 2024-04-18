% get raw EEG 
% modified from original code (preprocessed_data.m) 

clear
basedir = pwd;
stimulusdir = [basedir filesep 'stimuli'];
envelopedir = [stimulusdir filesep 'envelopes'];
if ~exist(envelopedir,'dir')
    mkdir(envelopedir);
end

% not all parameters are used
params.intermediatefs_audio = 8000; %Hz
params.envelopemethod = 'powerlaw';
params.subbandenvelopes = true;
params.subbandtag = ' subbands'; %if broadband, set to empty string: '';
params.spacing = 1.5;
params.freqs = erbspacebw(150,4000,params.spacing); % gammatone filter centerfrequencies
params.betamul = params.spacing*ones(size(params.freqs)); % multiplier for gammatone filter bandwidths
params.power = 0.3; % Powerlaw envelopes
params.intermediateSampleRate = 128; %Hz
params.lowpass_env = 9; % Hz, used for constructing a bpfilter used for both the audio and the eeg
params.lowpass = 60;
params.highpass = 1; % Hz
params.targetSampleRate = 128; % Hz
params.rereference = 'Cz';

% Build the bandpass filter
bpFilter = construct_bpfilter(params);
bpFilter_env = construct_bpfilter_env(params);
g = gammatonefir(params.freqs,params.intermediatefs_audio,[],params.betamul,'real'); % create real, FIR gammatone filters.% from amtoolbox>joergensen2011.m

%% Preprocess the audio files
stimulinames = list_stimuli();
nOfStimuli = length(stimulinames);
if 1
for i = 1:nOfStimuli
    % Load a stimulus
    [~,stimuliname,stimuliext] = fileparts(stimulinames{i});
    [audio,Fs] = audioread([stimulusdir filesep stimuliname stimuliext]);
    
    % resample to 8kHz 
    audio = resample(audio,params.intermediatefs_audio,Fs); 
    Fs = params.intermediatefs_audio;
    
    % Compute envelope
    if params.subbandenvelopes
        audio = real(ufilterbank(audio,g,1));
        audio = reshape(audio,size(audio,1),[]); 
    end
    
    % apply the powerlaw
    envelope = abs(audio).^params.power;
    
    % Intermediary downsampling of envelope before applying the more strict bpfilters
    envelope = resample(envelope,params.intermediateSampleRate,Fs);
    Fs = params.intermediateSampleRate;
    
    % bandpassilter the envelope
    envelope = filtfilt(bpFilter_env.numerator,1,envelope);
    
    % Downsample to ultimate frequency
    downsamplefactor = Fs/params.targetSampleRate;
    if round(downsamplefactor)~= downsamplefactor, error('Downsamplefactor is not integer'); end
    envelope = downsample(envelope,downsamplefactor);
    Fs = params.targetSampleRate;
    
    subband_weights = ones(1,size(envelope,2));
    % store as .mat files
    save([envelopedir filesep params.envelopemethod params.subbandtag ' ' stimuliname],'envelope','Fs','subband_weights');
    
end
end

%% Preprocess EEG and put EEG and corresponding stimulus envelopes together

preprocdir = [basedir filesep 'preprocessed'];
if ~exist(preprocdir,'dir')
    mkdir(preprocdir)
end
subjects = dir([basedir filesep 'S*.mat']);
% subjects = dir([basedir filesep 'S1.mat']);
subjects = sort({subjects(:).name});
postfix = '_dry.mat';


for subject = subjects
    load(fullfile(basedir,subject{1}))
    preproc_trials = {};
    
    Trials = {};
    Trials.eeg = {};
    Trials.envelope = {};
    Trials.wav = {};
    Trials.target_direction = [];
    Trials.target_gender = [];
    
    for trialnum = 1: size(trials,2) 
        
        trial = trials{trialnum};
        

        
        
        
        trial.FileHeader.SampleRate = params.targetSampleRate;
        
        % Load the correct stimuli, truncate to the length of EEG
        if trial.repetition,stimname_len = 16; else stimname_len = 12;end % rep_partX_trackX or partX_trackX
        
        %LEFT ear
        load([envelopedir filesep params.envelopemethod params.subbandtag ' ' trial.stimuli{1}(1:stimname_len) postfix ]);
        left = envelope(1:length(trial.RawData.EegData),:);
        [audio_left, fs_wav] = audioread(['stimuli' filesep  trial.stimuli{1}(1:stimname_len) '_dry.wav' ]);
        audio_left = resample(audio_left, 8000, 44100);
        audio_left = audio_left(1: ceil(length(trial.RawData.EegData) / 128 * 8000));
        
        %RIGHT ear
        load( [envelopedir filesep params.envelopemethod params.subbandtag ' ' trial.stimuli{2}(1:stimname_len) postfix ]);
        right = envelope(1:length(trial.RawData.EegData),:);
        [audio_right, fs_wav] = audioread(['stimuli' filesep  trial.stimuli{2}(1:stimname_len) '_dry.wav' ]);
        audio_right = resample(audio_right, 8000, 44100);
        audio_right = audio_left(1: ceil(length(trial.RawData.EegData) / 128 * 8000));
        
        
        trial.Envelope.AudioData = cat(3,left, right);
        trial.Envelope.subband_weights = subband_weights;
        trial.Wav.AudioData = cat(2, audio_left, audio_right);
        

        Trials.eeg{trialnum} = trial.RawData.EegData;
        Trials.envelope{trialnum} = squeeze(mean(cat(3,left, right), 2));
        Trials.wav{trialnum} = cat(2, audio_left, audio_right);
        if strcmp(trial.attended_ear, 'L')
            Trials.target_direction(trialnum) = 0;
        else
            Trials.target_direction(trialnum) = 1;
        end
        Trials.target_gender(trialnum) = 0;    % 0 for Das-2016

        
        
        preproc_trials{trialnum} = trial;
    end
    name_tmp = strsplit(subject{1}, '.');
    
    save(fullfile(preprocdir,[name_tmp{1,1} '_raw.mat']),'Trials')
end



function [ stimulinames ] = list_stimuli()
%List of stimuli names

stimulinames = {};

for experiment = [1 3]
    for track = 1:2
        if experiment == 1 % experiment 3 uses the same stimuli, but the attention of the listener is switched
            no_parts = 4;
            rep = false;
        elseif experiment ==3
            no_parts = 4;
            rep = true;
        end
        
        for part = 1:no_parts
            stimulinames =[stimulinames; {gen_stimuli_names(part,track,rep)}];
        end
    end
end
end

function [ filename ] = gen_stimuli_names(part,track,rep)
%Generates filename for audio stimuli

assert(islogical(rep));
assert(isnumeric(part));
assert(any(track == [1 2]));


part_tag = ['part' num2str(part)];
track_tag = ['track' num2str(track)];

cond_tag = 'dry';
extension = '.wav';

if rep == true
    rep_tag = 'rep';
elseif rep == false
    rep_tag = '';
end

separator = '_';
filename = [rep_tag separator part_tag separator track_tag separator cond_tag extension];
filename = regexprep(filename,[separator '+'],separator); %remove multiple underscores
filename = regexprep(filename,['^' separator],''); %remove starting underscore

end

function [ BP_equirip ] = construct_bpfilter( params )

Fs = params.intermediateSampleRate;
Fst1 = params.highpass-0.45;
Fp1 = params.highpass+0.45;
Fp2 = params.lowpass-0.45;
Fst2 = params.lowpass+0.45;
Ast1 = 20; %attenuation in dB
Ap = 0.5;
Ast2 = 15;
BP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2,Fs);
BP_equirip = design(BP,'equiripple');

end

function [ BP_equirip ] = construct_bpfilter_env( params )

Fs = params.intermediateSampleRate;
Fst1 = params.highpass-0.45;
Fp1 = params.highpass+0.45;
Fp2 = params.lowpass_env-0.45;
Fst2 = params.lowpass_env+0.45;
Ast1 = 20; %attenuation in dB
Ap = 0.5;
Ast2 = 15;
BP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2,Fs);
BP_equirip = design(BP,'equiripple');

end




