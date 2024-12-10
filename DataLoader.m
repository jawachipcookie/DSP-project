classdef DataLoader < handle
    properties
        data_path
        subject_id
        eeg_data
        epoch_indices
        features
    end
    
    methods
        function obj = DataLoader(data_path)
            obj.data_path = data_path;
            
            % Verify required files exist
            required_files = {'Acquisition 263 Data.cnt', ...
                            'epoch_inds.mat', ...
                            'all_features_ICA.mat'};
            obj.verify_files(required_files);
        end
        
        function verify_files(obj, required_files)
            for i = 1:length(required_files)
                if ~exist(fullfile(obj.data_path, required_files{i}), 'file')
                    error('Required file missing: %s', required_files{i});
                end
            end
        end
        
        function load_eeg_data(obj)
            % Load CNT file using EEGLAB
            try
                EEG = pop_loadcnt(fullfile(obj.data_path, 'Acquisition 263 Data.cnt'), ...
                    'dataformat', 'int32');
                obj.eeg_data = EEG.data;
                fprintf('Successfully loaded EEG data: %d channels, %d samples\n', ...
                    size(obj.eeg_data, 1), size(obj.eeg_data, 2));
            catch e
                error('Error loading CNT file: %s', e.message);
            end
        end
        
        function load_epochs(obj)
            % Load epoch indices
            epoch_data = load(fullfile(obj.data_path, 'epoch_inds.mat'));
            obj.epoch_indices = epoch_data.epoch_inds;
            fprintf('Loaded %d epochs\n', length(obj.epoch_indices));
        end
        
        function [epoched_data] = extract_epochs(obj, window_ms)
            % Extract epochs from continuous data
            if isempty(obj.eeg_data) || isempty(obj.epoch_indices)
                error('Load EEG data and epochs first');
            end
            
            % Convert ms to samples (assuming 1000Hz sampling rate)
            window_samples = round(window_ms);
            num_epochs = length(obj.epoch_indices);
            num_channels = size(obj.eeg_data, 1);
            
            % Initialize epoched data array
            epoched_data = zeros(num_epochs, num_channels, window_samples);
            
            % Extract each epoch
            for i = 1:num_epochs
                start_idx = obj.epoch_indices(i);
                end_idx = start_idx + window_samples - 1;
                
                if end_idx <= size(obj.eeg_data, 2)
                    epoched_data(i, :, :) = obj.eeg_data(:, start_idx:end_idx);
                else
                    warning('Epoch %d extends beyond data boundary', i);
                    % Pad with zeros if epoch extends beyond data
                    available_samples = size(obj.eeg_data, 2) - start_idx + 1;
                    epoched_data(i, :, 1:available_samples) = obj.eeg_data(:, start_idx:end);
                end
            end
        end
        
        function features = load_features(obj, feature_type)
            % Load pre-computed features
            switch feature_type
                case 'ica'
                    file_name = 'all_features_ICA.mat';
                case 'noica'
                    file_name = 'all_features_noICA.mat';
                case 'simple'
                    file_name = 'all_features_simple.mat';
                case 'regression'
                    file_name = 'regression_features.mat';
                case 'fft'
                    file_name = 'regression_features_FFT.mat';
                otherwise
                    error('Unknown feature type: %s', feature_type);
            end
            
            feature_data = load(fullfile(obj.data_path, file_name));
            % Get the first field name from the structure
            fields = fieldnames(feature_data);
            features = feature_data.(fields{1});
            fprintf('Loaded %s features: %dx%d\n', feature_type, size(features, 1), size(features, 2));
        end
    end
end