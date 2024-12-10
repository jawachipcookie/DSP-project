
% Load EEG Data and Preprocessing
EEG = pop_loadset('Acquisition 263 Data.set'); 
data_ica = load('all_features_ICA.mat'); 
data_noica = load('all_features_noICA.mat');
epochs = load('epoch_inds.mat');
fft_features = load('regression_features_FFT.mat');

% Display and visualize features
disp('ICA Data:');
disp(data_ica); % View loaded ICA data
plot(EEG.data(1, :)); % Plot raw EEG channel data
EEG = pop_eegfiltnew(EEG, 1, 40); % Bandpass filter (1–40 Hz)
EEG = pop_runica(EEG, 'extended', 1); % Perform ICA

% Extract Feature Matrix (example variable from loaded data)
feature_matrix = data_ica.feature_matrix; 
imagesc(feature_matrix); % Visualize the feature matrix as a heatmap

% Construct Adjacency Matrix
% Using a simple correlation matrix for adjacency as an example
adj_matrix = corr(feature_matrix'); 
imagesc(adj_matrix); % Visualize adjacency matrix

% Split Data into Training and Testing
labels = epochs.labels; % Assuming labels are in epoch_inds.mat
n_samples = size(feature_matrix, 1);
n_train = round(0.8 * n_samples); % 80% for training
n_test = n_samples - n_train;

train_data = feature_matrix(1:n_train, :);
train_labels = labels(1:n_train);
train_adj = adj_matrix(1:n_train, 1:n_train);

test_data = feature_matrix(n_train+1:end, :);
test_labels = labels(n_train+1:end);
test_adj = adj_matrix(n_train+1:end, n_train+1:end);

% Convert labels to one-hot encoding
train_labels_onehot = full(ind2vec(train_labels'))'; 
test_labels_onehot = full(ind2vec(test_labels'))';

% Parameters for GCN
input_dim = size(feature_matrix, 2); 
hidden_dim = 64; 
output_dim = size(train_labels_onehot, 2); 
learning_rate = 0.001;
epochs = 50;

% Initialize GCN Parameters
W1 = randn(input_dim, hidden_dim) * sqrt(2 / input_dim); % Layer 1 weights
W2 = randn(hidden_dim, output_dim) * sqrt(2 / hidden_dim); % Layer 2 weights
b1 = zeros(1, hidden_dim); 
b2 = zeros(1, output_dim);

% Activation Function
relu = @(x) max(0, x);

% Softmax Function
softmax = @(x) exp(x) ./ sum(exp(x), 2);

% Training the GCN
for epoch = 1:epochs
    % Forward Pass
    Z1 = train_data * W1 + b1; 
    H1 = relu(train_adj * Z1); % Graph convolution + ReLU
    Z2 = H1 * W2 + b2; 
    output = softmax(train_adj * Z2); % Final output

    % Compute Loss (Cross-Entropy)
    loss = -sum(sum(train_labels_onehot .* log(output + eps))) / size(train_data, 1);
    fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);

    % Backpropagation
    dout = output - train_labels_onehot; % Gradient of loss w.r.t output
    dW2 = H1' * (train_adj * dout);
    db2 = sum(train_adj * dout, 1);
    dH1 = dout * W2'; 
    dZ1 = (train_adj * dH1) .* (Z1 > 0); % Gradient w.r.t ReLU
    dW1 = train_data' * dZ1;
    db1 = sum(dZ1, 1);

    % Gradient Descent Update
    W1 = W1 - learning_rate * dW1;
    W2 = W2 - learning_rate * dW2;
    b1 = b1 - learning_rate * db1;
    b2 = b2 - learning_rate * db2;
end

% Testing the GCN
Z1_test = test_data * W1 + b1;
H1_test = relu(test_adj * Z1_test);
Z2_test = H1_test * W2 + b2;
output_test = softmax(test_adj * Z2_test);

% Predictions and Confusion Matrix
[~, pred] = max(output_test, [], 2);
[~, true_labels] = max(test_labels_onehot, [], 2);

conf_matrix = confusionmat(true_labels, pred);
accuracy = sum(pred == true_labels) / length(true_labels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
disp('Confusion Matrix:');
disp(conf_matrix);

% Visualize Results
figure;
subplot(1, 2, 1); 
imagesc(feature_matrix); title('Feature Matrix');
subplot(1, 2, 2); 
imagesc(adj_matrix); title('Adjacency Matrix');

