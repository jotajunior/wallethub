from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import random
from .parse_data import parse

def fit_model(x, y, x_test=None, y_test=None):
    alpha = 0.0001
    hidden_layer_sizes = (200, 200)
    activation = 'logistic'
    reg = MLPRegressor(hidden_layer_sizes, activation, 'adam', alpha, 
            max_iter=400)
    reg.fit(x, y)

    x_test = x_test if x_test else x
    y_test = y_test if y_test else y

    y_pred = reg.predict(x_test)
    result = [1 if abs(y_pred[i] - y_test[i]) <= 3.0 else 0 for i in range(len(y_pred))]
    
    accuracy = sum(result)/len(result)
    r2 = reg.score(x_test, y_test)

    return reg, accuracy, r2

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
    x, y = parse(filename)
    reg, accuracy, r2 = fit_model(x, y)

    joblib.dump(reg, dump_name)
