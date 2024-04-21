from tqdm import tqdm
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def dL_dW(W, x, y):
    num_data = len(x)

    derivative = 0
    for i in range(num_data):
        dLdz = dL_dz(W @ x[i], y[i])
        dzdW = dz_dW(W, x[i])
        dLdW = dLdz @ dzdW
        derivative += dLdW
    return - derivative / num_data

def dz_dW(W, xi):
    num_label = np.shape(W)[0]
    num_dim = len(xi)

    derivative = np.zeros((num_label, num_label, num_dim))
    for i in range(num_label):
        derivative[i, i, :] = xi
    return derivative

def dL_dz(zi, yi):
    # zi is 1 x data
    num_label = len(zi)
    indicator = np.zeros((num_label))
    indicator[yi] = 1
    return (-np.exp(zi)/ (np.sum(np.exp(zi))) + indicator)

def get_numerical_derivative(W, x, y, step):
    num_dim = np.shape(W)[1]
    num_label = np.shape(W)[0]

    derivative = np.zeros((num_label, num_dim))
    for k in range(num_label):
        for d in range(num_dim):
            W_step = np.copy(W)
            W_step[k, d] += step
            derivative[k, d] = (loss_function_all_data_debug(W_step, x, y) - loss_function_all_data_debug(W, x, y)) / step
    return derivative

class DataManager:
    def __init__(self, x, y, num_classes, ):
        self.num_classes = num_classes
        self.input_dim = np.shape(x)[1]
        self.x = x
        self.y = y
        self.num_data = len(x)

    def get_random_batch(self, batch_size):
        rand_int = np.random.randint(0, self.num_data, batch_size)
        return self.x[rand_int], self.y[rand_int]

class Model:
    def __init__(self, num_class, num_input_dim):
        self.num_class = num_class
        self.num_input_dim = num_input_dim
        self.W = np.random.rand(num_class, num_input_dim)

    def forward(self, x):
        # x is data x dims
        z = (self.W @ x.T).T # z is data x classes
        return self.softmax(z) # sm is data x classes

    def softmax(self, z):
        # z is data x classes
        numerator = np.exp(z)
        denominator = np.sum(np.exp(z), axis=1) # per data
        denominator = np.repeat(denominator[:, np.newaxis], axis=1, repeats=self.num_class) # data x classes
        return numerator / denominator

    def get_logits(self, sm):
        # sm is data x classes
        return np.argmax(sm, axis=1) # output is data x 1

def get_accuracy(preds, y):
    num_correct = np.sum(np.equal(preds, y))
    return num_correct / len(y)

def cross_entropy_loss(y_pred, y_true):
    # sm is data x classes
    # y is data x 1
    return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

def loss_function_all_data_debug(W, x, y):
    num_data = len(x)

    loss = 0
    for i in range(num_data):
        zi = W @ x[i]
        sm = np.exp(zi) / (np.sum(np.exp(zi)))
        log = np.log(sm)
        loss_i = log[y[i]]
        loss += loss_i
    loss = - np.mean(loss)
    return loss

def debug_loss_equivalent(model: Model, x, y):
    loss_nowmal = loss_function_all_data_debug(model.W, x, y)
    sm = model.forward(x)
    loss_model = cross_entropy_loss(sm, y)
    print(loss_nowmal)
    print(loss_model)


def optimize(model: Model, data_manager: DataManager, learning_rate, num_epochs, batch_size, momentum=0.0):
    num_iterations = int(data_manager.num_data / batch_size)
    accuracy_history = []
    gradient_with_momentum = np.zeros(model.W.shape)
    for e in tqdm(range(num_epochs)):
        loss = 0
        preds = []
        actual = []

        for i in range(num_iterations):
            x_batch, y_batch = data_manager.get_random_batch(batch_size)
            sm = model.forward(x_batch)
            gradient = dL_dW(model.W, x_batch, y_batch)
            gradient_with_momentum = (1 - momentum) * gradient + momentum * gradient_with_momentum
            model.W = model.W - learning_rate * gradient_with_momentum

            # Logging
            preds.append(model.get_logits(sm))
            actual.append(y_batch)
            loss += cross_entropy_loss(sm, y_batch) * batch_size

        print("Epoch:", e)
        preds = np.concatenate(preds)
        actual = np.concatenate(actual)
        accuracy = accuracy_score(actual, preds)
        print("Accuracy: ", accuracy)
        print("Loss: ", loss / data_manager.num_data)
        accuracy_history.append(accuracy)
    return accuracy_history


def debug_gradient_accuracy():
    num_label = 5
    num_data = 1000
    input_dimensions = 5
    y = np.random.randint(0, 5, num_data)
    x = np.random.rand(num_data, input_dimensions)
    x = np.concatenate([np.ones((num_data, 1)), x], axis=1)
    W = np.random.rand(num_label, input_dimensions+1)
    derivative = dL_dW(W, x, y)
    num_derivative = get_numerical_derivative(W, x, y, step=0.001)
    print("error", derivative - num_derivative)

def debug_train_test():
    num_label = 5
    num_data = 10000
    input_dimensions = 5
    # y = np.random.randint(0, 5, num_data)
    x = np.random.rand(num_data, input_dimensions)
    y = np.argmax(x, axis=1)
    x = np.concatenate([np.ones((num_data, 1)), x], axis=1)

    data_manager = DataManager(x=x, y=y, num_classes=num_label, batch_size=16)
    model = Model(num_class=num_label, num_input_dim=input_dimensions+1)
    # debug_loss_equivalent(model, x, y)
    optimize(model, data_manager, learning_rate=0.001, num_epochs=100)


def test_batch(data_manager):
    # Batch Size
    fig, ax = plt.subplots(figsize=(12, 8))

    momentum = 0.0
    lr = 1e-3
    batch_sizes = [1, 4, 16, 32]
    for batch_size in batch_sizes:
        model = Model(num_class=num_classes, num_input_dim=num_features)
        run = optimize(model, data_manager, learning_rate=lr, num_epochs=100, batch_size=batch_size,
                        momentum=momentum)
        ax.plot(np.arange(len(run)), run, linewidth=2,
                label='batch=' + str(batch_size) + ', lr=' + str(lr) + ', mom=' + str(momentum))
    # Set labels and title
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy v.s. Epoch')
    # Add a legend
    ax.legend()
    plt.show()

def test_lr(data_manager, momentum=0.0, batch_size=8):
    # Batch Size
    fig, ax = plt.subplots(figsize=(12, 8))

    lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    for lr in lrs:
        model = Model(num_class=num_classes, num_input_dim=num_features)
        run = optimize(model, data_manager, learning_rate=lr, num_epochs=100, batch_size=batch_size,
                        momentum=momentum)
        ax.plot(np.arange(len(run)), run, linewidth=2,
                label='batch=' + str(batch_size) + ', lr=' + str(lr) + ', mom=' + str(momentum))
    # Set labels and title
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy v.s. Epoch')
    # Add a legend
    ax.legend()
    plt.show()


def test_momentum(data_manager):
    # Batch Size
    fig, ax = plt.subplots(figsize=(12, 8))

    lr = 1e-3
    batch_size = 8
    momentums = [0.0, 0.5, 0.9, 0.99]
    for momentum in momentums:
        model = Model(num_class=num_classes, num_input_dim=num_features)
        run = optimize(model, data_manager, learning_rate=lr, num_epochs=100, batch_size=batch_size,
                        momentum=momentum)
        ax.plot(np.arange(len(run)), run, linewidth=2,
                label='batch=' + str(batch_size) + ', lr=' + str(lr) + ', mom=' + str(momentum))
    # Set labels and title
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy v.s. Epoch')
    # Add a legend
    ax.legend()
    plt.show()

if __name__=="__main__":
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

    # Convert y to integer values
    y = y.astype(int)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add bias term to the features
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    num_classes = 10
    num_features = X_train.shape[1]

    data_manager = DataManager(x=X_train, y=y, num_classes=num_classes)
    print("Num Data", data_manager.num_data)

    # test_batch(data_manager)
    test_lr(data_manager, batch_size=8, momentum=0.9)
    # test_momentum(data_manager)
