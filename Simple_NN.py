import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, der=False):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return sigmoid(x)*(1-sigmoid(x)) if der else 1 / (1 + np.exp(-x))


def relu(x, der=False):
    return x>0 if der else np.maximum(x, 0)

def tanh(x, der=False):
    return 1-(tanh(x))**2 if der else np.tanh(x)

def leaky_relu(x, der=False):
    l = 0.01
    return np.where(x>=0, 1, l) if der else np.where(x>=0, x, x*l)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100, af=relu, lr=0.1):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        np.random.seed(3023)
        self.num_step = num_step
        self.print_interval = print_interval
        self.af = af # activation function
        self.lr = lr # learning rate
        self.accuracy_box = []

        # Model parameters initialization
        # Please initiate your network parameters here.
        self.hidden1_weights = np.random.randn(2, hidden_size) # The dimension of the dataset is 2, so the initial dimension of the hidden weights is 2 x hidden size
        self.hidden2_weights = np.random.randn(hidden_size, hidden_size)
        self.output3_weights = np.random.randn(hidden_size, 1) # output is binary, so the output size is 1

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results. 

        Args:
            self.af: activation function
        """       
        self.z1 = self.af(np.dot(inputs, self.hidden1_weights)) 
        self.z2 = self.af(np.dot(self.z1, self.hidden2_weights))
        self.z3 = sigmoid(np.dot(self.z2, self.output3_weights)) # because the output interval must be [0, 1]
        return self.z3 # so the activation function of last layer must be sigmoid

    def backward(self, inputs):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.

        Args:
            self.af : activation function
            self.lr : learning rate
        """       
        self.error = self.error * sigmoid(self.output, der=True) # because the activation function of last layer must be sigmoid
        delta3_weights = np.dot(self.z2.T, self.error)

        self.error = np.dot(self.error, self.output3_weights.T) * self.af(self.z2, der=True) 
        delta2_weights = np.dot(self.z1.T, self.error)

        self.error = np.dot(self.error, self.hidden2_weights.T) * self.af(self.z1, der=True)
        delta1_weights = np.dot(inputs.T, self.error)

        self.hidden1_weights -= self.lr * delta1_weights
        self.hidden2_weights -= self.lr * delta2_weights
        self.output3_weights -= self.lr * delta3_weights

    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]

        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.forward(inputs[idx:idx+1, :])
                # print(idx, self.output)
                self.error = self.output - labels[idx:idx+1, :]
                self.backward(inputs[idx:idx+1, :])

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        accuracy = np.round((1 - error)*100, 3)
        self.accuracy_box.append(accuracy[0][0])
        print('accuracy: %.2f' % accuracy + '%')
        print('')


if __name__ == '__main__':
    seed = 3023
    np.random.seed(seed)
    data, label = GenData.fetch_data('Linear', 70)

    # af : activation function that you set
    # lr : learning that you set
    net = SimpleNet(100, num_step=2000, af=sigmoid, lr=0.1)
    accuracy_per_epoch = net.train(data, label)
    pred_result = np.round(net.forward(data))
    SimpleNet.plot_result(data, label, pred_result)
