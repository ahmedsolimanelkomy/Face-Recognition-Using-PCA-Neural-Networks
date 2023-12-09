%**** THIS FUNCTION CONTAIN THE DATA FEATURE EXTRACTION USING PCA ****

function [ projected_data ] = feature_extraction( data ,num_images )

    % Compute the mean face image
    mean_face = mean(data, 2);

    % Subtract the mean face from each image in the dataset
    centered_data = data - repmat(mean_face, 1, num_images);

    % Compute the covariance matrix of the centered data
    covariance_matrix = centered_data * centered_data' / (num_images - 1);

    % Compute the eigenvectors and eigenvalues of the covariance matrix
    [eigenvectors, eigenvalues ,latent] = pca(covariance_matrix);
    
    % Calculate the explained variance ratio
    explained_var = eigenvalues / sum(eigenvalues);
    cumulative_sum = cumsum(explained_var);
    
    % Plot the cumulative sum of the explained variance ratio
    figure;
    plot(cumulative_sum, 'LineWidth', 2);
    xlabel('Number of Principal Components');
    ylabel('Cumulative Sum of Explained Variance Ratio');
    title('Cumulative Sum of Explained Variance Ratio vs. Number of Principal Components');
    
    % Set the threshold for the minimum amount of variance to capture
    variance_threshold = 0.95;
    
    % Find the smallest number of principal components that captures at least the specified variance
    num_components = find(cumulative_sum >= variance_threshold, 1);
    
    % % Print the result
    disp(['Smallest number of principal components that captures at least 95% of the variance: ' num2str(num_components)]);
   
    % Select the top k eigenvectors that capture the most variance in the data
    k = 100;
    top_eigenvectors = eigenvectors(:, 1:k);

    % Project each image in the dataset onto the k eigenvectors
    projected_data = top_eigenvectors' * centered_data;

end

