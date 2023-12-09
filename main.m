clc 
clear all;

% % Setting the directory path where the images are stored
% data_dirpath ='C:\Users\ahmed\Desktop\g project\dataset';
% 
% % Reading Data and Labels
% [data , data_labels ,numt] = Read_Data(data_dirpath);
% 
% % Extract Features From Data Using Principle Component Analysis
% Training_Data_Features = feature_extraction(data',numt);
% 
% % applying cross validation to the dataset and labels
% [Xtrain,Ytrain,Xtest,Ytest] = cross_validation(Training_Data_Features ,data_labels);
% 
% % saving matrices
% save('Xtrain');
% save('Ytrain');
% save('Xtest');
% save('Ytest');

% loading matrices
Xtrain = load('Xtrain').Xtrain;
Ytrain = load('Ytrain').Ytrain;
Xtest = load('Xtest').Xtest;
Ytest = load('Ytest').Ytest;

% train feed forward neural network classifier
[net , tr] = FeedForwardNN(Xtrain',Ytrain',Xtest',Ytest');

% saving network matrices
save('net.mat');
save('tr.mat');
