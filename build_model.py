from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from parse_data import parse_training
from math import sqrt
import random

def fit_model(x, y, x_test=None, y_test=None):
    alpha = 0.0001
    hidden_layer_sizes = (200, 200)
    activation = 'logistic'
    reg = MLPRegressor(hidden_layer_sizes, activation, 'adam', alpha,
            max_iter=400)
    reg.fit(x, y)

    x_test = x_test if x_test != None else x
    y_test = y_test if y_test != None else y

    y_pred = reg.predict(x_test)
    result = [1 if abs(y_pred[i] - y_test[i]) <= 3.0 else 0 for i in range(len(y_pred))]
    
    accuracy = sum(result)/len(result)
    r2 = reg.score(x_test, y_test)

    return reg, accuracy, r2

def test_model_from_file(x, y, n=1, model_file='regressor.pkl'):
    reg = joblib.load(model_file)
    y_pred = reg.predict(x)

    result = [1 if abs(y_pred[i] - y[i]) <= 3.0 else 0 for i in range(len(y))]
    
    accuracy = sum(result)/len(result)
    r2 = reg.score(x, y)

    print('Results:')
    print('Accuracy: {}'.format(accuracy))
    print('R-squared: {}'.format(r2))
    print('% of std explained: {}'.format(1-sqrt(1-r2)))

def test_model(x, y, n=1):
    indexes = list(range(x.shape[0]))
    n_test = int(0.2*len(indexes))

    for i in range(n):
        random.shuffle(indexes)

        x_training = x[:n_test]
        y_training = y[:n_test]

        x_test = x[n_test:]
        y_test = y[n_test:]
        
        reg, accuracy, r2 = fit_model(
                x_training,
                y_training,
                x_test,
                y_test,
                )

        print('------- RUN #{} -------'.format(i+1))
        print('Accuracy: {}'.format(accuracy))
        print('R^2: {}'.format(r2))
        print('-----------------------')
        print()

def build_model(filename, dump_name='regressor.pkl'):
    x, y = parse_training(filename)
    reg, accuracy, r2 = fit_model(x, y)

    joblib.dump(reg, dump_name)

if __name__ == '__main__':
    x, y = parse_training('data.csv', n_imputations=10, poly=1, apply_pca=True)
    test_model(x, y, 5)
