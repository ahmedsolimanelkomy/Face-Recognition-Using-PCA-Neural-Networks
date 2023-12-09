%**** THIS FUNCTION APPLYING CROSS VALIDATION TO THE DATASET
function [Xtrain,Ytrain,Xtest,Ytest] = cross_validation(Training_Data_Features,data_labels)
    % Generate data
    X = Training_Data_Features'; % feature matrix
    Y = data_labels'; % label matrix
    
    % Define training and testing split
    trainPercent = 0.8;
    testPercent = 0.2;
    
    % Create cross-validation partition
    cv = cvpartition(size(X, 1), 'Holdout', testPercent);
    
    % Get the training and test indices
    trainIdx = cv.training;
    testIdx = cv.test;
    
    % Split the data into training and test sets
    Xtrain = X(trainIdx, :);
    Ytrain = Y(trainIdx, :);
    Xtest = X(testIdx, :);
    Ytest = Y(testIdx, :);

    % saving data matrices
    % save('Xtrain');
    % save('Ytrain');
    % save('Xtest');
    % save('Ytest');
end

