"""
Linear regression model to predict housing prices
How to run: python3 linearRegression.py
"""

import numpy as np

#  Load the data
#  Cross validation with an 80/20 split
def load_data():
    data = np.loadtxt('housing_HW6.csv', delimiter=',', dtype=float, skiprows = 1)

    # Shuffle the data
    np.random.shuffle(data)

    # Determine the index to split the data (80% training, 20% testing)
    split_index = int(0.8 * len(data))

    # Split the data into training and test
    training_data = data[:split_index, :]
    test_data = data[split_index:, :]

    # Separate x and y
    train_x = training_data[:, :-1]
    train_y = training_data[:, -1]  # Z

    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]
    return train_x, train_y, test_x, test_y

#  Add a column of ones to our features for the bias term (β0)
def add_ones(data):
    return np.c_[np.ones(data.shape[0]), data]  # H

#  Closed form solution to solving B^
def predict_features(H, train_y):
    H_T = np.transpose(H)
    return np.linalg.inv(H_T @ H) @ H_T @ train_y

def main():
     #  Load the data in
    train_x, train_y, test_x, test_y = load_data()

    #  First step: Choose parametric model f(x;B) and loss function L(B)
    #  f(x;B) = B0 + B1 * x[1] + ... + Bp * x[p]
    #  L(B) = mean((yi - f(xi;B))^2)

    # Add a column of ones to the features for the bias term (β0)
    H = add_ones(train_x)

    #  Second step: solve minimization of of L(B)
    #  Do this using the closed form solution
    #  B^ = (H^T * H)^-1 * H^T * Z
    beta_hat = predict_features(H, train_y)

    # Compute f(x;B)
    y_pred = H @ beta_hat

    #  Compute loss L(B) on training data, not needed since we are using closed form
    #loss = np.sum((train_y - y_pred)**2)

    #  Now apply B to predict our values for the test dataset.
    # H_Test = np.c_[np.ones(test_x.shape[0]), test_x]
    H_Test = add_ones(test_x)

    y_test_pred = H_Test @ beta_hat

    #  Compute the test and train MSE
    train_MSE = np.mean((train_y - y_pred)**2)
    test_MSE = np.mean((test_y - y_test_pred)**2)
    print(f"The MSE on the train dataset is {np.sqrt(train_MSE)}")
    print(f"The MSE on the test dataset is {np.sqrt(test_MSE)}")



if __name__ == "__main__":
    main()
