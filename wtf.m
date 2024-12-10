% Load EEG Dataset and Features
EEG = pop_loadset('Acquisition 263 Data.set');
data_ica = load('all_features_ICA.mat'); % Load ICA features
epochs = load('epoch_inds.mat'); % Load epoch indices



% Step 1: Prepare Feature and Adjacency Matrices
adjacency_matrix = data_ica.adj; % Adjacency matrix
feature_matrix = data_ica.feature_matrix; % Feature matrix (nodes x features)
labels = data_ica.labels; % Node labels
numNodes = size(feature_matrix, 1);

% Create Graph Representation
G = graph(adjacency_matrix); % Graph structure for visualization
figure; plot(G); title('Graph Structure of EEG Data');

% Step 2: Split Data into Training and Testing Sets
train_ratio = 0.8; % Split ratio
numTrainNodes = round(train_ratio * numNodes);
trainIdx = randperm(numNodes, numTrainNodes); % Randomly select training indices
testIdx = setdiff(1:numNodes, trainIdx); % Remaining nodes for testing

trainFeatures = feature_matrix(trainIdx, :);
trainLabels = labels(trainIdx, :);
testFeatures = feature_matrix(testIdx, :);
testLabels = labels(testIdx, :);

% Step 3: Define Graph Convolutional Network (GCN) Layers
numFeatures = size(feature_matrix, 2); % Number of input features
numClasses = numel(unique(labels)); % Number of output classes

layers = [
    imageInputLayer([numFeatures, 1, 1], 'Name', 'input') % Input layer for features
    fullyConnectedLayer(16, 'Name', 'fc1') % Simulated GCN Layer 1
    reluLayer('Name', 'relu1') % Activation layer 1
    fullyConnectedLayer(8, 'Name', 'fc2') % Simulated GCN Layer 2
    reluLayer('Name', 'relu2') % Activation layer 2
    fullyConnectedLayer(numClasses, 'Name', 'fc3') % Fully connected output layer
    softmaxLayer('Name', 'softmax') % Softmax for classification
    classificationLayer('Name', 'output') % Classification layer
];

% Step 4: Training Options for Neural Network
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testFeatures, categorical(testLabels)}, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Step 5: Train the Network
net = trainNetwork(trainFeatures, categorical(trainLabels), layers, options);

% Step 6: Evaluate the Model
% Predict labels using the trained network
predictedLabels = classify(net, testFeatures);

% Calculate Accuracy
accuracy = sum(predictedLabels == categorical(testLabels)) / numel(testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Generate and Display Confusion Matrix
figure;
confusionchart(categorical(testLabels), predictedLabels);
title('Confusion Matrix');

% Step 7: Save Results (Optional)
save('trained_network.mat', 'net');
save('adjacency_matrix.mat', 'adjacency_matrix');
