classdef Preprocessor < handle
    properties
        filter_low
        filter_high
        downsample_rate
    end
    
    methods
        function obj = Preprocessor()
            obj.filter_low = 0.1;  % Hz
            obj.filter_high = 50;  % Hz
            obj.downsample_rate = 4;  % Downsample from 1000Hz to 250Hz
        end
        
        function [processed_data] = process(obj, eeg_data)
            % Input: eeg_data (trials x channels x time)
            [num_trials, num_channels, num_samples] = size(eeg_data);
            processed_data = zeros(num_trials, num_channels, floor(num_samples/obj.downsample_rate));
            
            for trial = 1:num_trials
                % Get single trial data
                trial_data = squeeze(eeg_data(trial, :, :));
                
                % Apply bandpass filter
                filtered_data = obj.bandpass_filter(trial_data);
                
                % Downsample
                downsampled_data = downsample(filtered_data', obj.downsample_rate)';
                
                % Common Average Reference
                car_data = obj.common_average_reference(downsampled_data);
                
                % Store processed data
                processed_data(trial, :, :) = car_data;
            end
            
            % Z-score normalization across time
            processed_data = zscore(processed_data, 0, 3);
        end
        
        function filtered = bandpass_filter(obj, data)
            % Design bandpass filter
            nyquist = 500;  % Nyquist frequency (1000Hz/2)
            [b, a] = butter(4, [obj.filter_low obj.filter_high]/nyquist, 'bandpass');
            
            % Apply filter
            filtered = filtfilt(b, a, data')';
        end
        
        function car_data = common_average_reference(~, data)
            % Apply CAR
            car_data = data - mean(data, 1);
        end
    end
end