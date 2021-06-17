import pickle
import gzip
import numpy as np

def load_data():
    '''
 Возвращает кортеж из трех списков для данных обучения, проверки и тестирования соответственно. Каждый список содержит кортежи из двух элементов, где первый элемент представляет собой numpy ndarray из 784 (28X28 пикселей) строк и 1 столбец (вектор), а второй элемент - один hot-вектор, представляющий метку (0-9) (1 - если соответствует метке иначе 0). Выходные данные имеют размер 10 строк и 1 столбец, а также вектор.

Длина входных списков составляет 50000, 10000 и 10000, что соответствует количеству точек данных.
    '''    
    #Набор данных #MNIST
    with gzip.open('Documents/jupyter/lab-multi-layered-perceptron-nikitpronin-main 2/mnist.pkl.gz', 'rb') as f:
        tr_data, va_data, te_data = pickle.load(f,encoding="latin1")
    #список с конечными данными
    final_data = []
    #форматирование данных, для MLP
    for data in [tr_data, va_data, te_data]:
        #784 X 1 столбец данных на вход
        data_x = [x.reshape((784, 1)) for x in data[0]]
        data_y = np.array([x for x in data[1]])                    
        rows = data_y.shape[0]
        temp = np.zeros((rows, 10))
        #hot-вектора для numpy
        temp[np.arange(rows), data_y] = 1
        data_y = temp.tolist()
        #10 X 1 столбец данных на выход
        data_y = [np.array(y).reshape((10, 1)) for y in data_y]   
        #кортежи (x, y)               
        final_data.append(zip(data_x, data_y))        
    return (final_data[0], final_data[1], final_data[2])        
