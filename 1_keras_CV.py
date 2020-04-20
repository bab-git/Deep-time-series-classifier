"""

---------------------------------------------------------

*
Projekt: 			Hinkelstein
*
Abteilung: 			TSA-ES
*
Name:				1_Hinkelstein_keras_trainer
*
---------------------------------------------------------
*
Modul:
*
Version:			1.0.0

*
Autor: 				Dr.-Ing Pierre Gembaczka
*
Datum: 				03.04.2020
*
---------------------------------------------------------
*
Aenderungen: (neueste an oberster Stelle)
*
Version:			1.0.0
Datum:				03.04.2020
Verantwortlicher:	Dr.-Ing Pierre Gembaczka
Änderung:			Erste Implementierung
*
...
*
---------------------------------------------------------

"""
import matplotlib.pyplot as plt
import numpy as np
#import keyboard
np.random.seed()
from tensorflow import keras as ke
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from keras.callbacks import EarlyStopping
import pickle
from default_modules import *
import tensorflow as tf

from sklearn.svm import SVC


K = ke.backend

feat_dir = '/home/bhossein/BMBF project/Pierre_data/'

seed = int(input('seed'))

n_splits=5
n_repeats=5
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state = seed)
tp = np.zeros(n_splits*n_repeats)
fp = 100*np.ones(n_splits*n_repeats)
acc = np.zeros(n_splits*n_repeats)
#elapsed = np.empty(n_splits*n_repeats)
data = np.loadtxt(fname = feat_dir+"Features.txt")
(x, y_) = np.split(data, [13], axis=1)
(y_label,y_names) = np.split(y_, [2], axis=1)
y_label = y_label[:,0]+1


network_structure = [13,27,2]
n_epoch = 1000
verbose = 0
es_patience = 500
n_val =100
cv_select = 13
#%%
for cv_idx, (trn_idx, tst_idx) in enumerate(rskf.split(x, y_label)):
    if cv_idx > cv_select:
        break
    elif cv_idx != cv_select:
        continue
    
    print ('======= Traininf CV split: {}  rep: {}'.format(cv_idx % (n_splits)+1, np.floor(cv_idx/n_splits)+1))
    trn_idx, val_idx = train_test_split(trn_idx, test_size=len(tst_idx), stratify = y_label[trn_idx], random_state=seed)
      
    X_trn = x[trn_idx,:]    
    y_trn_with_names =  y_[trn_idx,:]
    
    X_test = x[tst_idx,:]
    y_test_with_names = y_[tst_idx,:]
    
    X_val = x[val_idx,:]
    val_data_with_names = y_[val_idx,:]

    (y_trn,y_trn_names) = np.split(y_trn_with_names, [2], axis=1)
    
    (y_test,y_test_names) = np.split(y_test_with_names, [2], axis=1)
    
    (y_val,y_val_names) = np.split(val_data_with_names, [2], axis=1)

    #X_trn, X_test, y_trn_with_names, y_test_with_names = train_test_split(x, y_, test_size=0.4, random_state= seed)
    
    #X_val, X_test, val_data_with_names, y_test_with_names = train_test_split(X_test, y_test_with_names, test_size=0.5, random_state=1)
    
    
    
    # -- network structure --
    #   13 input neurons
    #   27 one hidden layer with 3 neurons
    #   2 output neuron
    #12
    
    #tanh
    #26 oder 27 Neuronen / lr 0.1 / 93% - 16% / Test 40% Precicion 16
    
    model = ke.Sequential()
    model.add(ke.layers.Dense(network_structure[1], input_dim=network_structure[0], activation='softsign'))
    model.add(ke.layers.Dense(network_structure[2], activation='softmax'))

    # configure the optimizer
    opti = ke.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #opti = ke.optimizers.SGD(lr=0.2)
    
    
    # compile the model and use the mean_squared_error
    #categorical_crossentropy
    #mean_squared_error
    model.compile(loss='mean_squared_error',
                  optimizer=opti,
                  metrics=['accuracy'])


    # Use early stopping -> If the loss on the testdata stops increasing, the training is finished.
    es = ke.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
    
    #model.load_weights("13_27_2_p40_softsign.h5")
    
    # training
#    model.fit(X_trn, y_trn,validation_data=(X_test, y_test),
#              epochs=n_epoch,batch_size=len(X_trn), verbose = verbose, callbacks = [es])
    
    
    model.fit(X_trn, y_trn,
              epochs=n_epoch,batch_size=len(X_trn), verbose=verbose)

    #%%------
    print("============== Val data loop")

    for i in range(n_val):
        pred = model.predict(X_test)
    
        richtig = 0
    
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
    
        first_step = 0
    
        for i in range(len(y_test)):
            if y_test[i][0] == 1 and pred[i][0]> 0.5:
                richtig = richtig + 1
                true_neg = true_neg + 1
            if y_test[i][1] == 1 and pred[i][1]> 0.5:
                richtig = richtig + 1
                true_pos = true_pos + 1
            if y_test[i][0] == 1 and pred[i][1] > 0.5:
                false_pos = false_pos + 1
            if y_test[i][1] == 1 and pred[i][0] > 0.5:
                false_neg = false_neg + 1
    
        true_pos_percent = true_pos * 100 / (true_pos + false_neg)
        false_pos_percent = false_pos * 100 / (false_pos + true_neg)
    
        #if keyboard.is_pressed('a'):
            #break
#        print("============== Val data loop")
#        print(true_pos_percent)
#        print(false_pos_percent)
        if  true_pos_percent >= 93 and  false_pos_percent <= 18:
            break
        else:
            if true_pos_percent >= 90.0 and  false_pos_percent <= 20: #89.2 Sigmoid / 87 RELU
                #K.set_value(model.optimizer.lr, .01)
                model.fit(X_trn, y_trn, epochs=500,batch_size=len(X_trn), verbose=0)
#                model.fit(X_trn, y_trn, validation_data=(X_test, y_test), epochs=500,batch_size=len(X_trn), verbose=verbose)
                first_step = 1
            else:
                model.fit(X_trn, y_trn, epochs=1000,batch_size=len(X_trn), verbose=0)
#                model.fit(X_trn, y_trn, validation_data=(X_test, y_test), epochs=1000,batch_size=len(X_trn), verbose=verbose)
    
    #model.fit(X_trn, y_trn,validation_data=(X_test, y_test),epochs=20000,batch_size=len(X_trn), verbose=2,callbacks=[es])
    print(true_pos_percent)
    print(false_pos_percent)
    #%%
    print("============== Test data")
    
    #model.save_weights("13_27_2_p40_softsign.h5")
    
    #write_data_training = np.concatenate((X_trn, y_trn_with_names), axis=1)
    #write_data_test = np.concatenate((X_test, y_test_with_names), axis=1)
    
    #np.savetxt('training_data.txt', write_data_training, fmt='%1.16f',delimiter='\t')
    #np.savetxt('test_data.txt', write_data_test, fmt='%1.16f',delimiter='\t')
    
    # predict on test
#    del X_test, y_test
    pred = model.predict(X_val)
    
    #print(pred)
    
    richtig = 0
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for i in range(len(y_val)):
        if y_val[i][0] == 1 and pred[i][0]> 0.5:
            richtig = richtig + 1
            true_neg = true_neg + 1
        if y_val[i][1] == 1 and pred[i][1]> 0.5:
            richtig = richtig + 1
            true_pos = true_pos + 1
        if y_val[i][0] == 1 and pred[i][1] > 0.5:
            false_pos = false_pos + 1
        if y_val[i][1] == 1 and pred[i][0] > 0.5:
            false_neg = false_neg + 1
    
    tp[cv_idx] = (true_pos * 100 / (true_pos + false_neg))
    fp[cv_idx] = (false_pos * 100 / (false_pos + true_neg)) 
    acc[cv_idx] = (richtig * 100) / len(y_val)
    
    print(str(true_pos) + " True positive / " + str(true_pos * 100 / (true_pos + false_neg)) + " %" )
    print(str(false_pos) + " False positive / " + str(false_pos * 100 / (false_pos + true_neg)) + " %" )
    #for i in range(len(y_val)):
    #    print(str(y_val[i][0]) + " - " + str(pred[i][0]) + " / " + str(y_val[i][1]) + " - " + str(pred[i][1]))
#%% 
if cv_select != None:
    assert 1==2    
#print(str(len(y_val)) + " total")
#print(str(richtig) + " correct")
#print(str((richtig * 100) / len(y_val)) + " % correct")
#print(str(true_pos + false_neg) + " positive Datasets")
#print(str(true_neg + false_pos) + " negative Datasets")
#print(str(true_pos) + " True positive / " + str(true_pos * 100 / (true_pos + false_neg)) + " %" )
#print(str(false_pos) + " False positive / " + str(false_pos * 100 / (false_pos + true_neg)) + " %" )
#print(str(true_neg) + " True negative")
#print(str(false_neg) + " False negative")
print("{:>40}  {:.2f}".format("Min test accuracy:", acc.min()))
print("{:>40}  {:.2f}".format("Max test accuracy:", acc.max()))
print("{:>40}  {:.2f}".format("Mean test accuracy:", acc.mean()))
print("{:>40}  {:.2f}".format("Test accuracy standard deviation:", acc.std()))

print("{:>40}  {:.2f}".format("Min TP rate:", tp.min()))
print("{:>40}  {:.2f}".format("Max TP rate:", tp.max()))
print("{:>40}  {:.3f}".format("Mean TP rate:", tp.mean()))
print("{:>40}  {:.2f}".format("TP rate standard deviation:", tp.std()))

print("{:>40}  {:.2f}".format("Min FP rate:", fp.min()))
print("{:>40}  {:.2f}".format("Max FP rate:", fp.max()))
print("{:>40}  {:.3f}".format("Mean FP rate:", fp.mean()))
print("{:>40}  {:.2f}".format("FP rate standard deviation:", fp.std()))

cv_acc = {'tp':tp, 'fp':fp, 'acc':acc}
pickle.dump(cv_acc,open(result_dir+"temp.p","wb"))
#pickle.dump(cv_acc,open("train_pirere_CV_total.p","wb"))
#pickle.dump(cv_acc,open("train_pirere_CV_total_es.p","wb"))
#%%
plt.close('all')
plt.figure()
plt.boxplot([acc, tp, fp], labels=["Accuracy", "True Positives", "False Positives"], showmeans=True)
plt.title("{} splits, {} repeats".format(n_splits, n_repeats))
plt.grid()
plt.show()
plt.yticks(ticks=list(range(0,100,10)))

#%% confidence
X_pred = X_val.copy()
y_pred = y_val.copy()

y_pred = y_pred[:,1]
pred = model.predict(X_pred)

#print(pred)

richtig = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

true_pos = ((pred[:,1] > pred[:,0]) & (1 == y_pred)).sum().item()
total_P = (1 == y_pred).sum().item()

false_pos = ((pred[:,1] > pred[:,0]) & (0 == y_pred)).sum().item() 
false_pos_idx = np.where(((pred[:,1] > pred[:,0]) & (0 == y_pred)))[0]
total_N = len(y_pred)-total_P

false_neg = ((pred[:,0] > pred[:,1]) & (1 == y_pred)).sum().item() 
false_neg_idx = np.where(((pred[:,0] > pred[:,1]) & (1 == y_pred)))[0]


#for i in range(len(y_pred)):
#    if y_val[i][0] == 1 and pred[i][0]> 0.5:
#        richtig = richtig + 1
#        true_neg = true_neg + 1
#    if y_val[i][1] == 1 and pred[i][1]> 0.5:
#        richtig = richtig + 1
#        true_pos = true_pos + 1
#    if y_val[i][0] == 1 and pred[i][1] > 0.5:
#        false_pos = false_pos + 1
#    if y_val[i][1] == 1 and pred[i][0] > 0.5:
#        false_neg = false_neg + 1

#tp_rate = (true_pos * 100 / total_P)
#fp_rate = (false_pos * 100 / total_N) 

tp_rate = (true_pos * 100 / (total_P))
fp_rate = (false_pos * 100 / (total_N)) 

print(tp_rate)
print(fp_rate)
#acc[cv_idx] = (richtig * 100) / len(y_val)

y_rel = np.array([0 if i in np.append(false_pos_idx,false_neg_idx) else 1 
                  for i in range(len(y_pred))])
s_neg = (y_rel == 0).sum()
    
clf = SVC(gamma='auto')
clf.fit(X_pred, y_rel)
conf_pred = clf.predict(X_pred)
((y_rel == 0) & (conf_pred == 0)).sum()/(y_rel == 0).sum()