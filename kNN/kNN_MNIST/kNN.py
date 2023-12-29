"""
k-Nearest neighbors implementation for the MNIST data set
How to run: python3 kNN.py
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#  Load the data
#  Cross validation with an 80/20 split
def load_data():
    mat_train_x = sio.loadmat('MNIST_train_image.mat') 
    mat_train_y = sio.loadmat('MNIST_train_label.mat') 
    mat_test_x = sio.loadmat('MNIST_test_image.mat') 
    mat_test_y = sio.loadmat('MNIST_test_label.mat') 
    train_img = mat_train_x['trainX']
    train_label = mat_train_y['trainL']
    test_img = mat_test_x['testX']
    test_label = mat_test_y['testL']

    # # Shuffle the data, turned this off since data is being saved
    # np.random.shuffle(data)

    # Determine the index to split the data (50,000 training, 10,000 test)
    split_index = 50000

    train_img_set, val_img_set = train_img[:, :split_index], train_img[:, split_index:]
    train_label_set, val_label_set = train_label[:split_index], train_label[split_index:]

    return train_img_set.T, train_label_set.flatten(), val_img_set.T, val_label_set.flatten(), test_img.T, test_label.flatten()

def compute_knn(train_x, train_y, test_x, test_y, K_set):
    test_size = test_x.shape[0]

    # Code adapted from ece499_HW_6 Solutions
    y_hat_test = np.zeros((test_size, K_set.size), dtype=int)
    #  store the accuracy per k_value here, this is to measure validation accuracy
    accuracies = np.zeros((K_set.size,), dtype=int)

    # The (i,k) entry of y_hat_test will save the K-NN decision for the i-th
    # test data point when the k-th entry of K_set is used as K for the K-NN
    # approach. 
    for i in range(test_size):
        # print(f"Loop {i} out of {test_size}")
        x = test_x[i]

        # Compute the distances from x to training data points
        dist = np.sum((train_x - x) ** 2, axis=1)

        # Compute decisions of K-NN (for values K)
        for k in range(K_set.size):
            K = K_set[k]
            idx = np.argpartition(dist, K)

            # Get nearest labels of neighboring K digits
            nearest_labels = train_y[idx[:K]]

            # Use bincount along with np.argmax to find the most frequent label
            y_hat_test[i][k] = np.argmax(np.bincount(nearest_labels))

            # Update validation accuracies (if y_hat_test doesnt match test_y, we add 1 to the accuracy)
            accuracies[k] += int(y_hat_test[i][k] != test_y[i])

    return accuracies



def main():
    #  k values to be tested and plotted
    k_values = np.array([1, 5, 10, 20, 50, 100, 200, 500]).astype(int)

    #  Load the data in
    train_img_set, train_label_set, val_img_set, val_label_set, test_img, test_label = load_data()
    
    # File paths for saving/loading kNN results
    kNN_validation_file = 'kNN_validation.npy'
    kNN_training_file = 'kNN_training.npy'
    kNN_test_file = 'kNN_test.npy'

    # Try to load kNN results from files
    try:
        kNN_validation = np.load(kNN_validation_file)
        kNN_training = np.load(kNN_training_file)
        kNN_test = np.load(kNN_test_file)
        print("kNN results loaded from files.")
    except FileNotFoundError:
        # If files not found, compute kNN
        kNN_validation = compute_knn(train_img_set, train_label_set, val_img_set, val_label_set, k_values)
        kNN_training = compute_knn(train_img_set, train_label_set, train_img_set, train_label_set, k_values)
        kNN_test = compute_knn(train_img_set, train_label_set, test_img, test_label, k_values)

        # Save kNN results
        np.save(kNN_validation_file, kNN_validation)
        np.save(kNN_training_file, kNN_training)
        np.save(kNN_test_file, kNN_test)

        print("kNN results computed and saved.")

    # Plot the training and test MSE versus K
    plt.plot(k_values, np.divide(kNN_training, train_img_set.shape[0]), label='Training Error Rate')
    plt.plot(k_values, np.divide(kNN_test, test_img.shape[0]), label='Test Error Rate')
    plt.plot(k_values, np.divide(kNN_validation, val_img_set.shape[0]), label='validation Error Rate')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('(Error Rate)')
    plt.title('Error Rate versus K')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()