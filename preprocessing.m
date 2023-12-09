%**** THIS SCRIPT CONTAIN THE DATA PREPROCESSING TECHNIQUES ****
clc 
clear all;

% Load the face images
faceDir = 'C:\Users\ahmed\Desktop\g project\TrainDatabase';
faceFiles = dir(fullfile(faceDir, '*.jpg'));
numFaces = length(faceFiles);

% Loop through each face image
for i = 1:numFaces

    % Load the current face image
    filename = fullfile(faceDir, faceFiles(i).name);
    img = imread(filename);

    % Convert the image to grayscale
    grayImg = im2gray(img);
    grayImg = double(grayImg);

    % Normalize the image using mean and standard deviation
    meanVal = mean(grayImg(:));
    stdVal = std(grayImg(:));
    normalizedImg = (grayImg - meanVal) / stdVal;

    % Apply a Gaussian filter to remove noise
    filteredImg = imgaussfilt(normalizedImg, 2);

    % Rotate the image by a random angle between -10 and 10 degrees
    angle = randi([-1, 1]);
    rotatedImg = imrotate(filteredImg, angle, 'crop');

    % Flip the image horizontally with a 50% probability
    flipProb = rand();
    if flipProb > 0.5
        flippedImg = flip(rotatedImg, 2);
    else
        flippedImg = rotatedImg;
    end

    % Save the augmented image to disk
    path = 'C:\Users\ahmed\Desktop\g project\dataset';
    [~ ,name, ext] = fileparts(filename);
    outputFilename = fullfile(path, [name '_aug' ext]);
    imwrite(flippedImg, outputFilename);

end