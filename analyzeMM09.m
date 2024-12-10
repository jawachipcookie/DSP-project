% Set up the path
data_path = fullfile('C:', 'Users', 'Vishesh', 'Documents', 'MATLAB', 'MM09');

% Verify the path exists
if ~exist(data_path, 'dir')
    error('Directory not found: %s', data_path);
end

% Initialize data loader
try
    loader = DataLoader(data_path);
    fprintf('Successfully initialized DataLoader for path: %s\n', data_path);
catch e
    error('Error initializing DataLoader: %s', e.message);
end

% Load the data
try
    % Load raw EEG dataloader
    loader.load_eeg_data();
    
    % Load epoch information
    loader.load_epochs();
    
    % Extract epochs (2000ms window)
    epoched_data = loader.extract_epochs(2000);
    
    % Load different types of features
    fprintf('\nLoading features...\n');
    ica_features = loader.load_features('ica');
    simple_features = loader.load_features('simple');
    
    fprintf('\nData loading complete!\n');
    fprintf('EEG data size: %d channels x %d timepoints\n', size(loader.eeg_data));
    fprintf('Number of epochs: %d\n', size(epoched_data, 1));
    fprintf('ICA features size: %d x %d\n', size(ica_features));
    
catch e
    fprintf('Error during data loading: %s\n', e.message);
    fprintf('Error occurred in: %s\n', e.stack(1).name);
end