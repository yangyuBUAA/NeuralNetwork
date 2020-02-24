import math
import numpy as np


class ActivateFunction:
    """
    activate function class
    """
    def __init__(self, act_name):
        self.activateName = act_name

    def activate(self, x):
        if self.activateName == 'sigmoid':
            return 1/(1 + math.pow(math.e, -x))
        elif self.activateName == 'ReLU':
            return max(x, 0)
        elif self.activateName == 'Tanh':
            pass

    def derivate(self, x):
        if self.activateName == 'sigmoid':
            return (1/(1 + math.pow(math.e, -x))) * (1-(1/(1 + math.pow(math.e, -x))))
        elif self.activateName == 'ReLU':
            if x > 0:
                return 1
            else:
                return 0
        elif self.activateName == 'Tanh':
            pass


class LossFunction:
    """
    loss function class
    """
    def __init__(self, loss_name):
        self.lossName = loss_name

    def calculate(self, predict_y, y):
        if self.lossName == 'MSE':
            result = (1/2) * math.pow((predict_y-y), 2)
        elif self.lossName == 'logLoss':
            pass
        elif self.lossName == 'cross':
            pass

    def derivate(self, predict_y, y):
        if self.lossName == 'MSE':
            return predict_y - y
        elif self.lossName == 'logLoss':
            pass
        elif self.lossName == 'cross':
            pass


class Tools:
    """
    Tools class
    """
    @staticmethod
    def apply_func_for_ndarray(func, np_array):
        """
        used to apply function to ndarray's every element
        :param func: function
        :param np_array: input ndarrray
        :return: output ndarray
        """
        arr_list = list(np_array)
        calculate_list = []
        for each in arr_list:
            calculate_list.append(func(each))
        np_array_calculate = np.array(calculate_list)
        return np_array_calculate.reshape(-1, 1)

    @staticmethod
    def apply_func_for_2_ndarray(func, np_array_1, np_array_2):
        """
        used to apply calculate for two ndarray's every element
        :param func: calculate function
        :param np_array_1: ndarray 1
        :param np_array_2: ndarray 2
        :return: result ndarray
        """
        arr1_list = list(np_array_1)
        arr2_list = list(np_array_2)
        calculate_list = []
        for ind in range(len(arr1_list)):
            calculate_list.append(func(arr1_list[ind], arr2_list[ind]))
        np_array_calculate = np.array(calculate_list)
        return np_array_calculate.reshape(-1, 1)

    @staticmethod
    def train_set_batch_split(train, batch_size):  # has not been finished!!!!!!!!!!
        data_size = train.shape[0]
        print(data_size)
        print(batch_size)
        if data_size % batch_size != 0:
            batch_num = data_size / batch_size
        else:
            batch_num = data_size / batch_size
        data_set = []
        for ind in range(3):
            data_set.append(train[ind:ind+batch_size])
        print(data_set)
        return data_set


class FeedNeuralNetwork:
    def __init__(self, num_layers, hidden_per_layers):
        """
        initial network private variables
        :param num_layers: neural network layers
        :param hidden_per_layers: neural networks layers params
        """
        self._num_layers = num_layers
        self._hidden_per_layers = hidden_per_layers
        # initialize model parameters, random_initialize_params
        self._weight = []  # size: numlayers -1
        self._bias = []  # size: numlayers -1
        self.random_initialize_params()  # initial weight matrix and bias vector
        # forward propagation result, forward
        self._forward_z = []  # size:numlayers -1
        self._forward_a = []  # size:numlayers -1
        # backpropagation error, backward
        self._back_error = []  # size:numlayers -1
        # calculate derivation accumulate, calculate_delta
        self._accumulate_weight = []  # size:numlayers -1
        self._accumulate_bias = []  # size:numlayers -1

    def random_initialize_params(self):
        """
        initial model parameters
        :return:None
        """
        for ind in range(self._num_layers - 1):  # initialize weight and bias
            matrix = np.random.random((self._hidden_per_layers[ind+1], self._hidden_per_layers[ind]))
            bias = np.random.random((self._hidden_per_layers[ind+1],1))
            self._weight.append(matrix)
            self._bias.append(bias)

    def show_model_params(self):
        """
        print model parameters
        :return:None
        """
        print('network layers: ' + str(self._num_layers))
        print('network hidden neural nums per layer: ')
        print(self._hidden_per_layers)
        for ind in range(self._num_layers - 1):
            print('network weight parameters: ' + str(ind+1) + ' layer ')
            print('shape: ', self._weight[ind].shape)
            print(self._weight[ind])
            print('network bias parameters: ' + str(ind+1) + ' layer')
            print('shape: ', self._bias[ind].shape)
            print(self._bias[ind])

    def forward(self, x, activate_func):
        """
        forward calculate, saved calculate process to self._forward_z, self._forward_a
        :param x:
        :param activate_func:
        :return:
        """
        self._forward_z = []
        self._forward_a = []
        # print(x.shape)
        x = x.reshape(-1, 1)  # transfer to vector
        # print(x.shape)
        activateF = ActivateFunction(activate_func)
        hidden_1_z = np.dot(self._weight[0], x) + self._bias[0]
        hidden_1_a = Tools.apply_func_for_ndarray(activateF.activate, hidden_1_z)
        self._forward_z.append(hidden_1_z)
        self._forward_a.append(hidden_1_a)
        for ind in range(self._num_layers - 2):
            z = np.dot(self._weight[ind + 1], self._forward_a[ind])
            a = Tools.apply_func_for_ndarray(activateF.activate, z)
            self._forward_z.append(z)
            if ind == self._num_layers - 3:
                self._forward_a.append(z)
            else:
                self._forward_a.append(a)
        # print(self._forward_a)
        # print(self._forward_z)

    def backward(self, y, activate_func, loss_func):
        self._back_error = []
        y = y.reshape(-1, 1)
        lossF = LossFunction(loss_func)
        activateF = ActivateFunction(activate_func)
        predict_y = self._forward_a[-1]
        predict_z = self._forward_z[-1]
        loss = Tools.apply_func_for_2_ndarray(lossF.calculate, predict_y, y)
        loss_derivate_predict_y = Tools.apply_func_for_2_ndarray(lossF.derivate, predict_y, y)
        # predict_y_derivate_z = Tools.apply_func_for_ndarray(activateF.derivate, predict_z)
        # output_error = loss_derivate_predict_y * predict_y_derivate_z
        output_error = loss_derivate_predict_y
        self._back_error.insert(0, output_error)
        weight_ind = self._num_layers - 2
        for ind in range(self._num_layers-2):
            present_error = np.dot(self._weight[weight_ind].T, self._back_error[0])
            weight_ind = weight_ind - 1
            self._back_error.insert(0, present_error)
        # print('error')
        # for each in self._back_error:
        #     print(each.shape)

    def calculate_delta(self, x):
        x = x.reshape(1, -1)
        # print(self._back_error[0].shape)
        accumulate_weight_1 = np.dot(self._back_error[0], x)
        accumulate_bias_1 = self._back_error[0]
        if len(self._accumulate_weight) == 0:
            self._accumulate_weight.append(accumulate_weight_1)
            self._accumulate_bias.append(accumulate_bias_1)
            for ind in range(self._num_layers - 2):
                accumulate_weight_present = np.dot(self._back_error[ind + 1], self._forward_a[ind].reshape(1, -1))
                accumulate_bias_present = self._back_error[ind + 1]
                self._accumulate_weight.append(accumulate_weight_present)
                self._accumulate_bias.append(accumulate_bias_present)
        else:
            self._accumulate_weight[0] = self._accumulate_weight[0] + accumulate_weight_1
            self._accumulate_bias[0] = self._accumulate_bias[0] + accumulate_bias_1
            for ind in range(self._num_layers - 2):
                accumulate_weight_present = np.dot(self._back_error[ind + 1], self._forward_a[ind].reshape(1, -1))
                accumulate_bias_present = self._back_error[ind + 1]
                self._accumulate_weight[ind + 1] = self._accumulate_weight[ind + 1] + accumulate_weight_present
                self._accumulate_bias[ind + 1] = self._accumulate_bias[ind + 1] + accumulate_bias_present

        # print('accumulate_weight')
        # for each in self._accumulate_weight:
        #     print(each)
        # print('accumulate_bias')
        # for each in self._accumulate_bias:
        #     print(each)

    def update_params(self, learning_rate):
        for ind in range(self._num_layers - 1):
            self._weight[ind] = self._weight[ind] - learning_rate * self._accumulate_weight[ind]
            self._bias[ind] = self._bias[ind] - learning_rate * self._accumulate_bias[ind]

        self._accumulate_weight = []
        self._accumulate_bias = []

    def train(self, train, learning_rate, activate_func, loss_func, batch_size, epoch_size):
        """
        network's training function
        :param train: train_Set(np.ndarray)
        :param learning_rate:
        :param activate_func:
        :param loss_func:
        :param batch_size:
        :param epoch_size:
        :return:
        """
        train_set = Tools.train_set_batch_split(train, batch_size)
        for epoch in range(epoch_size):
            print('epoch: ' + str(epoch))
            for each in train_set:  # train_set has been transformed to many batch
                x = each[:, :-1]
                y = each[:, -1]
                for line in range(x.shape[0]):
                    self.forward(x[line], activate_func)  # reset self._forward, then calculate forward params
                    self.backward(y[line], activate_func, loss_func)  # reset self._back_error,then calculate error
                    self.calculate_delta(x[line])  # calculate batch_size gradient'summary
                self.update_params(learning_rate)  # update network parameters,then reset self._accumulate

    def test(self):
        self.forward(np.array([0.1, 0.3, 0.7]), 'ReLU')
        print(self._forward_a)


class DatasetGeneration:
    @staticmethod
    def generation():
        data_list = []
        for x in np.arange(0, 1, 0.1):
            y = math.pow(x, 2)
            data_list.append([x, y])
        return np.array(data_list)


if __name__ == '__main__':
    train_set = np.array([[0.1, 0.3, 0.7, 1.5], [0.2, 0.5, 0.1, 1.5], [0.8, 0.2, 0.5, 1]])
    net = FeedNeuralNetwork(4, [3, 3, 3, 1])
    net.show_model_params()
    net.train(train_set, 0.1, 'ReLU', 'MSE', 1, 10000)
    net.show_model_params()
    net.test()

    # Tools.train_set_batch_split(train_set, 1)
    # train_set = np.array([[1, 2, 3, 1], [1, 6, 7, 1], [1, 9, 8, 1]])
    # # print(train_set.shape)
    # train_x = train_set[:, :3]
    # train_y = train_set[:, 3].reshape(-1, 1)
    # # print(train_x.shape, train_y.shape)

    # net.forward(train_x[0], 'sigmoid')
    # net.backward(train_y[0], 'sigmoid', 'MSE')
    # net.calculate_delta(train_x[0])
    # net.show_model_params()
    # net.update_params(0.5)
    # net.show_model_params()
    # DatasetGeneration.generation()

