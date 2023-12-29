"""
Tips: Use logscale while plotting lambda (lambda vs error rates)
validation dataset is used to determine best lambda value
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

    # # Shuffle the data
    # np.random.shuffle(data)

    # Determine the index to split the data (50,000 training, 10,000 test)
    split_index = 50000

    train_img_set, val_img_set = train_img[:, :split_index], train_img[:, split_index:]
    train_label_set, val_label_set = train_label[:split_index], train_label[split_index:]

    return train_img_set.T, train_label_set.flatten(), val_img_set.T, val_label_set.flatten(), test_img.T, test_label.flatten()


def gradient_descent(X_train, y_train, lambd, step_size):
    print(f"Training with reglarization parameter lamda: {lambd}")
    # Initialize parameters
    num_classes = 10  # Number of classes (digits 0 to 9)
    num_features = X_train.shape[1]  # Number of features (pixels in this case)
    learning_rate = step_size
    epochs = 3000

    # Convert labels to one-hot encoding, IE: matrix of 0's and 1's where y_i = k_bar
    y_train_one_hot = np.eye(num_classes)[y_train]

    # Initialize beta
    beta = np.zeros((num_classes, (num_features + 1)))

    # Add a column of ones to the features
    X_train_ones = np.insert(X_train, 0, 1, axis=1)

    # matrix to hold the loss, will be plotted as iterations vs loss
    iter = np.zeros((epochs // 100) + 1)
    loss_arr = np.zeros((epochs // 100) + 1)
    i = 0
    # Gradient Descent
    for epoch in range(epochs + 1):
        # Forward pas
        logits = beta @ X_train_ones.T
        probabilities = np.exp(logits) /  np.sum(np.exp(logits), axis=0, keepdims=True)

        # Compute cross-entropy loss L(B)
        regularized_beta = beta**2
        loss = -np.mean(np.sum(y_train_one_hot * np.log(probabilities.T) + lambd * np.sum(regularized_beta), axis=1))

        # Backward pass
        gradient = -( 1 / X_train.shape[0]) * (X_train_ones.T @ (y_train_one_hot - probabilities.T)) + 2 * lambd * beta.T

        # Update beta
        beta = beta - learning_rate * gradient.T

        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            iter[i] = epoch
            loss_arr[i] = loss
            i += 1
            print(f"Epoch {epoch}, Loss: {loss}")

    #  Plot the loss vs iterations
    plt.plot(iter, loss_arr, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('loss vs iterations')
    plt.legend()
    plt.show()

    return beta

def predict_label(test_x, beta):
    #  Predict the label using the trained beta value

    # Add a column of ones to the features
    test_x_ones = np.insert(test_x, 0, 1, axis=1)

    # Apply parameters to make the prediction
    predict = beta @ test_x_ones.T
    # Predict the class with the highest probability for each example
    predictions = np.argmax(predict, axis=0)
    return predictions
    

def calculate_accuracy(predictions, labels):
    #  Obtain accuracy of trained data
    return np.mean(predictions == labels)


def main():
    # a) Description of parametric model assumption:
    #    P(Y = k | X = x) for k in {1,2,...,C}
    #    Find equation in slide 3 of lecture 21

    # b) Loss function with regularization lambda L(B)
    #    L(B) = -1 * sum_n(sum_c(1{y=k} * ln [P(Y=k | X = x_i; B)]))
    #    Where n is the number of training points, C is the number of features (ie: 0-9)
    #  Load the data in
    train_img_set, train_label_set, val_img_set, val_label_set, test_img, test_label = load_data()

    lambda_set = [0.0001, 0.001, 0.01, 0.1, 0, 1] # as lambda increases, decrease the step size
    step_set = [0.2, 0.02, 0.002, 0.0001, 0.2, 0.0001]

    beta = []

    try:
        # Load all beta values into an array for later access
        for i in range(6):
            beta.append(np.load(f'trained_beta_{lambda_set[i]}.npy'))
        print("Trained beta values loaded!")

    except OSError as e:
        print(f"No trained beta found, training beta now...")
        for i in range(len(lambda_set)):
            beta_trained = gradient_descent(train_img_set, train_label_set, lambda_set[i], step_set[i])
            np.save(f'trained_beta_{lambda_set[i]}', beta_trained)
            beta.append(beta_trained)

        print(f"All beta values trained, calculating accuracies")

    # This code obtains the test accuracy for the best label, only used this for the homework document
    # predictions_test = predict_label(test_img, beta[0])
    # accuracy_test = np.mean(predictions_test == test_label)
    # print(f"Accuracy on test with lambda {lambda_set[0]}: {accuracy_test * 100:.2f}%")

    # predictions_test = predict_label(test_img, beta[4])
    # accuracy_test = np.mean(predictions_test == test_label)
    # print(f"Accuracy on test with lambda {lambda_set[4]}: {accuracy_test * 100:.2f}%")

    # Collect error rates
    train_errors = []
    val_errors = []
    test_errors = []

    for i in range(len(beta)):
        predictions_train = predict_label(train_img_set, beta[i])
        accuracy_train = calculate_accuracy(predictions_train, train_label_set)
        train_errors.append(1 - accuracy_train)

        predictions_validation = predict_label(val_img_set, beta[i])
        accuracy_validation = calculate_accuracy(predictions_validation, val_label_set)
        print(f"Accuracy on validation with lambda {lambda_set[i]:<6} and step size {step_set[i]:<7} is: {accuracy_validation * 100:.2f}%")
        val_errors.append(1 - accuracy_validation)

        predictions_test = predict_label(test_img, beta[i])
        accuracy_test = calculate_accuracy(predictions_test, test_label)
        test_errors.append(1 - accuracy_test)

    #  Since we are log scaling our plot, plotting log(0) will create issues, set lambda 0 to 0.1
    for i in range(len(lambda_set)):
        if lambda_set[i] == 0:
            lambda_set[i] += 0.1
    
    # lambda_set = [0.0001, 0.001, 0.01, 0.1, 0.01, 1]
    # Plotting
    plt.plot(lambda_set, train_errors, label='Training Error')
    plt.plot(lambda_set, val_errors, label='Validation Error')
    plt.plot(lambda_set, test_errors, label='Test Error')

    plt.xlabel('Lambda')
    plt.ylabel('Error Rate')
    plt.xscale('log')  # Use log scale for lambda values
    plt.title('Error Rate vs. Lambda')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()