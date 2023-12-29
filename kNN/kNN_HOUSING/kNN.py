"""
k-Nearest neighbors implementation for the housing data set
How to run: python3 kNN.py
"""

import numpy as np
import matplotlib.pyplot as plt

#  Load the data
#  Cross validation with an 80/20 split
def load_data():
    data = np.loadtxt('housing_data.csv', delimiter=',', dtype=float, skiprows = 1)

    # Shuffle the data
    np.random.shuffle(data)

    # Determine the index to split the data (80% training, 20% testing)
    split_index = int(0.8 * len(data))

    # Split the data into training and test
    training_data = data[:split_index, :]
    test_data = data[split_index:, :]

    # Separate x and y
    train_x = training_data[:, :-1]
    train_y = training_data[:, -1]

    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]
    return train_x, train_y, test_x, test_y

#  Normalize both training and test x data
def normalize_data(train_x, test_x):
    #  normalize the data, first calculate the mean of each datapoint
    mu = np.mean(train_x, axis=0)
    sigma = np.std(train_x, axis=0)
    #  normalized training adn test data
    train_norm = (train_x - mu) / sigma
    test_norm = (test_x - mu) / sigma
    return train_norm, test_norm

#  Compute knn on x values, return training mse
#  Train norm: normalized training data to make predictions with
#  Test norm: normalized data that we are trying to predict
def compute_knn(k_values, train_norm, test_norm, train_y, test_y):
    test_mse_values = []
    #  Compute the kNN decision using K-nearest neighbors
    for k in k_values:
        print(f"knn with {k} neighbors processing...")
        # Initialize an array to store predictions
        predictions = []

        # For each k, make a prediction on test
        for i in range(len(test_norm)):
            #  Calculuate euclidean distance for each sample
            distances = np.linalg.norm(train_norm - test_norm[i], axis=1)

            #  get the indices of nearest k neighbors
            k_nearest_idx = np.argsort(distances)[:k]

            #  Get labels of those neighbors
            k_nearest_labels = train_y[k_nearest_idx]

            #  Make a prediction
            prediction = np.mean(k_nearest_labels)

            #  save predictions to the l
            predictions.append(prediction)

        #  Calculate the accuracy
        predictions = np.array(predictions)

        # Calculate MSE for test set
        test_mse = np.mean((predictions - test_y)**2)
        test_mse_values.append(test_mse)
        
        print(f"knn with {k} neighbors processed!")

    return test_mse_values

def plot(k_values, train_mse_values, test_mse_values):
    # Plot the training and test MSE versus K
    plt.plot(k_values, train_mse_values, label='Training MSE')
    plt.plot(k_values, test_mse_values, label='Test MSE')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training and Test MSE versus K')
    plt.legend()
    plt.show()

def main():
    #  k values to be tested and plotted
    k_values = [1, 5, 10, 20, 50, 100, 200]

    #  Load the data in
    train_x, train_y, test_x, test_y = load_data()

    #  Normalize the data
    train_norm, test_norm = normalize_data(train_x, test_x)

    #  Calculate MSE on nearest neighbors

    # Store MSE values to be plotted (based on k values)
    print(f"\nNow computing MSE values with knn on test data...\n")
    test_mse_values = compute_knn(k_values, train_norm, test_norm, train_y, test_y)  # MSE on test values

    print(f"\n\nNow computing MSE values with knn on training data...")
    train_mse_values = compute_knn(k_values, train_norm, train_norm, train_y, train_y)  # MSE on training values

    #  Plot the values
    plot(k_values, train_mse_values, test_mse_values)

if __name__ == "__main__":
    main()
