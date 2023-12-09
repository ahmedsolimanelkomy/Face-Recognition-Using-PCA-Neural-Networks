%**** THIS FUNCTION CONTAIN FEED FORWARD NEURAL NETWORK CLASSIFIER ****

function [ net , tr ] = FeedForwardNN( Training_Features , Training_Labels ,Testing_Features,Testing_Lables)
        
    % Define the network architecture

        hidden_size = 20; % number of hidden units
        output_size = 5; % number of output size(number of classes)
        net = patternnet(hidden_size); %define the network

        % set up network parameters and train the network on the training set:
        net.trainFcn = 'traingdx'; % Gradient descent with momentum and adaptive learning rate backpropagation
        net.trainParam.max_fail = 10; % maximum number of validation failures
        net.trainParam.epochs = 1000; % maximum number of training epochs
        net.performFcn = 'mse';
        net.trainParam.goal = 0.001; % training goal (mean squared error)
        net.performParam.regularization = 0.1;

        % specify network layers activation functions
        net.layers{1}.transferFcn = 'logsig'; % sigmoid activation function for hidden layer neurons
        net.layers{end}.transferFcn = 'softmax'; % softmax activation function for output layer neurons (to obtain class probabilities)
        
        % Set the number of neurons in the output layer
        net.layers{end}.size = output_size;
        
        % data division
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 0.7;
        net.divideParam.testRatio = 0.15; 
        net.divideParam.ValRatio = 0.15; 

    % Train the network

        trl = dummyvar(categorical(Training_Labels')); % convert training labels to categorical labels
        trl = trl';
        tsl = dummyvar(categorical(Testing_Lables')); % convert testing labels to categorical labels
        tsl = tsl';
        [net, tr] = train(net, Training_Features, trl);

    % Test the Network
        y = net(Training_Features);
        e = gsubtract(Training_Labels,y);
        performance = perform(net,Training_Labels,y);
        tind = vec2ind(Training_Labels);
        yind = vec2ind(y);
        percentErrors = sum(tind ~= yind)/numel(tind);
        disp(['percentErrors: ' num2str(percentErrors) '%']);

        % Recalculate Training, Validation and Test Performance
        trainTargets = Training_Labels .* tr.trainMask{1};
        valTargets = Training_Labels .* tr.valMask{1};
        testTargets = Training_Labels .* tr.testMask{1};
        
        trainPerformance = perform(net,trainTargets,y);
        disp(['trainPerformance: ' num2str(trainPerformance)]);
        valPerformance = perform(net,valTargets,y);
        disp(['valPerformance: ' num2str(valPerformance)]);
        testPerformance = perform(net,testTargets,y);
        disp(['testPerformance: ' num2str(testPerformance)]);

        % Simulate neural network with testing data
        predicted_labels = sim(net, Testing_Features);
        
        % Convert output to categorical labels
        predicted_labels = vec2ind(predicted_labels);
        
        % Compute accuracy and confusion matrix
        conf_matrix = confusionmat(Testing_Lables, predicted_labels);
        accuracy = sum(diag(conf_matrix))/sum(conf_matrix(:))*100;
        disp(['Accuracy: ' num2str(ceil(accuracy)) '%']);
        
        % Display confusion chart patternnet
        figure;
        confusionchart(Testing_Lables, predicted_labels);

        % Compute precision and recall
        num_classes = size(conf_matrix, 1);
        precision = zeros(num_classes, 1);
        recall = zeros(num_classes, 1);
        for i = 1:num_classes
            TP = conf_matrix(i,i);
            FP = sum(conf_matrix(:,i))-TP;
            FN = sum(conf_matrix(i,:))-TP;
            TN = sum(conf_matrix(:))-TP-FP-FN;
            precision(i) = TP / (TP + FP);
            recall(i) = TP / (TP + FN);
        end
        
        % Plot precision and recall
        figure;
        bar(precision);
        hold on;
        bar(recall);
        xlabel('Class');
        ylabel('Precision/Recall');
        legend('Precision', 'Recall');


        % Compute the MSE for the training, validation, and testing subsets
        train_mse = perform(net, trl(:, tr.trainInd), net(Training_Features(:, tr.trainInd)));
        val_mse = perform(net, trl(:, tr.valInd), net(Training_Features(:, tr.valInd)));
        test_mse = perform(net, tsl, net(Testing_Features));
        
        % Display the MSE values
        disp(['Training MSE: ' num2str(train_mse)]);
        disp(['Validation MSE: ' num2str(val_mse)]);
        disp(['Testing MSE: ' num2str(test_mse)]);

end

