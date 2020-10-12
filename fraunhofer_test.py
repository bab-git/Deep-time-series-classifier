import os
import numpy as np
#import keyboard
np.random.seed()
from tensorflow import keras as ke
from sklearn.model_selection import train_test_split

def evaluate(X_test, y_test, model_name, model_dir, conf_thresh=0.5, print_results=True):

    # -- network structure --
    #   13 input neurons
    #   27 one hidden layer with 3 neurons
    #   2 output neuron
    #12
    network_structure = [13,27,2]

    #tanh
    #26 oder 27 Neuronen / lr 0.1 / 93% - 16% / Test 40% Precicion 16

    model = ke.Sequential()
    model.add(ke.layers.Dense(network_structure[1], input_dim=network_structure[0], activation='softsign'))
    model.add(ke.layers.Dense(network_structure[2], activation='softmax'))

    # configure the optimizer
    opti = ke.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #opti = ke.optimizers.SGD(lr=0.2)

    model.compile(loss='mean_squared_error',
                optimizer=opti,
                metrics=['accuracy'])

    model.load_weights(os.path.join(model_dir, model_name + ".h5"))

    # predict
    pred = model.predict(X_test)

    print(pred)

    richtig = 0

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    below_thresh = []

    for i in range(len(y_test)):
        if y_test[i][0] == 1 and pred[i][0]> conf_thresh:
            richtig = richtig + 1
            true_neg = true_neg + 1
        elif y_test[i][1] == 1 and pred[i][1]> conf_thresh:
            richtig = richtig + 1
            true_pos = true_pos + 1
        elif y_test[i][0] == 1 and pred[i][1] > conf_thresh:
            false_pos = false_pos + 1
        elif y_test[i][1] == 1 and pred[i][0] > conf_thresh:
            false_neg = false_neg + 1
        else:
            below_thresh.append(i)


    for i in range(len(y_test)):
        print(str(y_test[i][0]) + " - " + str(pred[i][0]) + " / " + str(y_test[i][1]) + " - " + str(pred[i][1]))

    acc = richtig #* 100 / len(y_test)
    tp = true_pos #* 100 / (true_pos + false_neg)
    fp = false_pos #* 100 / (false_pos + true_neg)

    if print_results:
        print(str(len(y_test)) + " total")
        print(str(richtig) + " correct")
        print(str(acc) + " % correct")
        print(str(true_pos + false_neg) + " positive Datasets")
        print(str(true_neg + false_pos) + " negative Datasets")
        print(str(true_pos) + " True positive / " + str(tp) + " %" )
        print(str(false_pos) + " False positive / " + str(fp) + " %" )
        print(str(true_neg) + " True negative")
        print(str(false_neg) + " False negative")

    return tp, fp, acc, below_thresh