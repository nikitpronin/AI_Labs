import numpy as np
import random
import time
import data_loader
import sys

class Network:
    '''
    Класс нейросети.
    '''
    def __init__(self, neuron_list, no_inputs):
        #количество нейронов для каждого слоя 
        self.neuron_list = neuron_list
        self.no_of_layers = len(neuron_list)
        #weight[i] сохраняет веса между слоем i и слоем i + 1. 
        #веса инициализируются из нормального распределения
        self.weights = [np.random.normal(0, np.sqrt(2./no_inputs), (n0*n1)).reshape(n1, n0) for n0, n1 in zip(neuron_list[:-1], neuron_list[1:])]
        #смещения сохраняются начиная с первого скрытого слоя и далее 
        self.biases = [np.random.normal(0, np.sqrt(2./no_inputs), neurons).reshape(neurons, 1) for neurons in neuron_list[1:]]


    def feedforward(self, z):    
        '''
        Формирует список для MLP по входному вектору z
        '''  
        z_list = []
        z_list.append(z)
        for i in xrange(self.no_of_layers - 1):
            #z - это вход для слоя i, biases [i] - это вектор смещения для слоя i + 1.            
            z = self.weights[i].dot(z) + self.biases[i]
            z_list.append(z)
            a = sigmoid(z)
            z = a
        return z_list

    
    def stochastic_gradient_descent(self, tr_data, no_epochs, mini_batch_size, eta, test_data = None):
        '''
        Выполняет стохастический градиентный спуск и вызывает обратное распространение для обучения нейронной сети. 
        '''
        for i in xrange(no_epochs):
            start = time.time()
            #рандомим данные для стохастической выборки 
            random.shuffle(tr_data)
        
            mini_batches = [tr_data[j : j+mini_batch_size] for j in xrange(0, len(tr_data), mini_batch_size)]
            for mini_batch in mini_batches:
                #тренируем каждый пакет данных со скоростью обучения eta 
                self.update_network(mini_batch, eta, mini_batch_size)            
            print ('Epoch {} complete'.format(i))
            if(test_data is not None):
                print ('Evaluation results: {}/{}'.format(self.evaluate(test_data), len(test_data)))
            end = time.time()
            print ('Time taken: {}s'.format(end - start))


    def update_network(self, mini_batch, eta, mini_batch_size):
        #nabla w и nabla b хранят текущую сумму, полученную в результате обратного распространения ошибки 
        #для всех данных в мини-пакете 
        nabla_w_sum = []
        nabla_b_sum = []
        for x, y in mini_batch:
            #шаг обучения
            nabla_w_x, nabla_b_x = self.back_propogation(x, y)
            #записываем сумму
            if(len(nabla_w_sum) == 0 and len(nabla_b_sum) == 0):
                nabla_w_sum = nabla_w_x
                nabla_b_sum = nabla_b_x
            else:
                nabla_w_sum = [x+y for x, y in zip(nabla_w_sum, nabla_w_x)]
                nabla_b_sum = [x+y for x, y in zip(nabla_b_sum, nabla_b_x)]                                       
        #обновляем веса
        self.weights = [weight - ((eta/mini_batch_size) * nabla_w_i) for weight, nabla_w_i in zip(self.weights, nabla_w_sum)]
        self.biases = [bias - ((eta/mini_batch_size) * nabla_b_i) for bias, nabla_b_i in zip(self.biases, nabla_b_sum)]    


    def back_propogation(self, x, y):
        z_list = self.feedforward(x)   
        a_list = [z_list[0]] + [sigmoid(z) for z in z_list[1:-1]] 
        #вычисляем ошибку на последнем слое
        output_error = (sigmoid(z_list[-1]) - y) * sigmoid_prime(z_list[-1])               
        error = [output_error]
        #распространение ошибки на предыдущие слои 
        for i in xrange(self.no_of_layers - 2, 0, -1):            
            err =  self.weights[i].transpose().dot(error[0]) * sigmoid_prime(z_list[i])            
            error = [err] + error
        #вычисляем изменение веса из-за ошибки
        nabla_w_x = [err.dot(a.transpose()) for err, a in zip(error, a_list)]
        
        return nabla_w_x, error
        
    
    def evaluate(self, test_data):       
        #считаем количество правильных прогнозов
        return sum([(np.argmax(sigmoid(self.feedforward(x)[-1])) == np.argmax(y)) for (x, y) in test_data])
    

#Вспомогательные функции
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))        


if __name__ == '__main__':
    #выводим весь массив numpy, этим можно принебречь, но полезно во время отладки
    np.set_printoptions(threshold=np.inf)  
    tr_data, val_data, test_data = data_loader.load_data()       
    neuron_list = sys.argv[1][1:-1].split(',')
    neuron_list = [int(neurons) for neurons in neuron_list]
    n = Network(neuron_list, len(tr_data))
    n.stochastic_gradient_descent(tr_data, int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), test_data)
