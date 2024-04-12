import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_loss_derivative(W, x, y):
    num_data = len(x)

    derivative = 0
    for i in range(num_data):
        dLdz = log_der_matrix_i(W @ x[i], y[i])
        dzdW = matrix_derivative_i(W, x[i])
        dLdW = dLdz @ dzdW
        derivative += dLdW
    return - derivative / num_data

def matrix_derivative_i(W, xi):
    num_label = np.shape(W)[0]
    num_dim = len(xi)

    derivative = np.zeros((num_label, num_label, num_dim))
    for i in range(num_label):
        derivative[i, i, :] = xi
    return derivative

def indicator_i(num_label, yi):
    indicator = np.zeros((num_label))
    indicator[yi] = 1
    return indicator

def log_der_matrix_i(zi, yi):
    # zi is 1 x data
    num_label = len(zi)
    # multiplier = np.exp(zi[yi]) / (np.sum(np.exp(zi)))**2
    # multiplier = 1 / (np.sum(np.exp(zi)))
    return (-np.exp(zi)/ (np.sum(np.exp(zi))) + indicator_i(num_label, yi))

def get_numerical_derivative(W, x, y, step):
    num_dim = np.shape(W)[1]
    num_label = np.shape(W)[0]

    derivative = np.zeros((num_label, num_dim))
    for k in range(num_label):
        for d in range(num_dim):
            W_step = np.copy(W)
            W_step[k, d] += step
            derivative[k, d] = (loss_function(W_step, x, y) - loss_function(W, x, y)) / step
    return derivative

class DataManager:
    def __init__(self, x, y, num_classes, batch_size):
        self.num_classes = num_classes
        self.input_dim = np.shape(x)[1]
        self.x = x
        self.y = y
        self.num_data = len(x)
        self.batch_size = batch_size

    def get_random_batch(self):
        rand_int = np.random.randint(0, self.num_data, self.batch_size)
        return self.x[rand_int], self.y[rand_int]

class Model:
    def __init__(self, num_class, num_input_dim):
        self.num_class = num_class
        self.num_input_dim = num_input_dim
        self.W = np.random.rand(num_label, num_input_dim)

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

def get_loss(sm, y):
    # sm is data x classes
    # y is data x 1
    num_data = len(y)
    data_indices = np.arange(num_data)
    log = np.log(sm)
    loss = log[data_indices, y]
    loss = np.sum(loss)
    return - loss / num_data

def loss_function(W, x, y):
    num_data = len(x)

    loss = 0
    for i in range(num_data):
        zi = W @ x[i]
        sm = np.exp(zi) / (np.sum(np.exp(zi)))
        log = np.log(sm)
        loss_i = log[y[i]]
        loss += loss_i
    loss = - loss / num_data
    return loss

def debug_loss_equivalent(model: Model, x, y):
    loss_nowmal = loss_function(model.W, x, y)
    sm = model.forward(x)
    loss_model = get_loss(sm, y)
    print(loss_nowmal)
    print(loss_model)



def optimize(model: Model, data_manager: DataManager, learning_rate, num_epochs):
    num_iterations = int(data_manager.num_data / data_manager.batch_size)
    accuracy_history = []
    for e in tqdm(range(num_epochs)):
        loss = 0
        preds = []
        actual = []
        for i in range(num_iterations):
            x_batch, y_batch = data_manager.get_random_batch()
            sm = model.forward(x_batch)
            derivative = get_loss_derivative(model.W, x_batch, y_batch)
            model.W = model.W - learning_rate * derivative

            # Logging
            preds.append(model.get_logits(sm))
            actual.append(y_batch)
            loss += get_loss(sm, y_batch)

        print("Epoch:", e)
        preds = np.concatenate(preds)
        actual = np.concatenate(actual)
        accuracy = get_accuracy(preds, actual)
        print("Accuracy: ", accuracy)
        print("Loss: ", loss / data_manager.num_data)
        accuracy_history.append(accuracy)
    plt.plot(accuracy_history)
    plt.show()


def debug_gradient_accuracy(x, y):
    W = np.random.rand(num_label, input_dimensions+1)
    # print(W)
    derivative = get_loss_derivative(W, x, y)
    # print(derivative)
    num_derivative = get_numerical_derivative(W, x, y, step=0.001)
    # print(num_derivative)
    print("error", derivative - num_derivative)

if __name__=="__main__":
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

    # debug_gradient_accuracy(x, y)