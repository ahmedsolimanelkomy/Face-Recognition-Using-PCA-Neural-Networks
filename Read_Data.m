%**** THIS FUNCTION CONTAIN THE DATA READING  TO A PROPER SHAPE  ****

function [ Data ,labels ,numImages] = Read_Data(dirPath)

        % Get the list of image files in the directory
        imageFiles = dir(fullfile(dirPath, '*.jpg'));
        numImages = numel(imageFiles);
        labels = zeros(numImages,1);
        % Read the images into a 3D matrix
        Data = zeros(64, 64, length(imageFiles));
        for i = 1:length(imageFiles)
            image = fullfile(dirPath, imageFiles(i).name);
            a =imread(image);
            image = im2gray(a);
            Data(:, :, i) = double(imresize(image,[64 64]));
            label = strtok(imageFiles(i).name, '_'); % extract the label from the filename
            labels(i) = str2double(label); % store the label in the cell array
        end

        % convert labels to a row vector
        labels = labels';

        % Reshape the 3D matrix into a 2D matrix
        Data = reshape(Data,64*64, length(imageFiles));

        % normalizaion of images values
        Data = double(Data')/255;
                
end
    