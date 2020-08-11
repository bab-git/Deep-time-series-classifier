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
import pandas as pd

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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
cv_select = 3
#%%
for cv_idx, (trn_idx, tst_idx) in enumerate(rskf.split(x, y_label)):
    if cv_idx > cv_select:
        break
    elif cv_idx != cv_select:
        continue
    
    print ('======= Training CV split: {}  rep: {}'.format(cv_idx % (n_splits)+1, np.floor(cv_idx/n_splits)+1))
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
#    es = ke.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
    
    #model.load_weights("13_27_2_p40_softsign.h5")
    
    # training
#    model.fit(X_trn, y_trn,validation_data=(X_val, y_val),
#              epochs=n_epoch,batch_size=len(X_trn), verbose = verbose, callbacks = [es])
    
    
    model.fit(X_trn, y_trn,
              epochs=n_epoch,batch_size=len(X_trn), verbose=verbose)

    #%%------
    print("============== Val data loop")

    for i_val in range(n_val):
        pred = model.predict(X_val)
    
        richtig = 0
    
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
    
        first_step = 0
    
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
                model.fit(X_trn, y_trn, epochs=500,batch_size=len(X_trn), verbose = verbose)
#                model.fit(X_trn, y_trn, validation_data=(X_val, y_val), epochs=500,batch_size=len(X_trn), verbose=verbose)
                first_step = 1
            else:
                model.fit(X_trn, y_trn, epochs=1000,batch_size=len(X_trn), verbose = verbose)
#                model.fit(X_trn, y_trn, validation_data=(X_val, y_val), epochs=1000,batch_size=len(X_trn), verbose=verbose)
    
    #model.fit(X_trn, y_trn,validation_data=(X_val, y_val),epochs=20000,batch_size=len(X_trn), verbose=2,callbacks=[es])
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
    pred = model.predict(X_test)
    
    #print(pred)
    
    richtig = 0
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
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
    
    tp[cv_idx] = (true_pos * 100 / (true_pos + false_neg))
    fp[cv_idx] = (false_pos * 100 / (false_pos + true_neg)) 
    acc[cv_idx] = (richtig * 100) / len(y_test)
    
    print(str(true_pos) + " True positive / " + str(true_pos * 100 / (true_pos + false_neg)) + " %" )
    print(str(false_pos) + " False positive / " + str(false_pos * 100 / (false_pos + true_neg)) + " %" )
    #for i in range(len(y_test)):
    #    print(str(y_test[i][0]) + " - " + str(pred[i][0]) + " / " + str(y_test[i][1]) + " - " + str(pred[i][1]))
#%% 
if cv_select != None:
    assert 1==2    
#print(str(len(y_test)) + " total")
#print(str(richtig) + " correct")
#print(str((richtig * 100) / len(y_test)) + " % correct")
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
#pickle.dump(cv_acc,open(result_dir+"train_pirere_CV_correct.p","wb"))
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
#model = tf.keras.models.load_model(result_dir+'IMS_model_worse')

c_thresh = 0.95

X_pred = X_val.copy()
y_pred = y_val.copy()


pred = model.predict(X_pred)

result = np.append(pred, y_pred, axis= 1)
y_pred = y_pred[:,1]

#print(pred)

richtig = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

richtig_idx = np.where(((pred[:,0] > pred[:,1]) & (0 == y_pred)) | ((pred[:,1] > pred[:,0]) & (1 == y_pred)) )[0]
richtig_vec = np.zeros(len(y_pred))
richtig_vec[richtig_idx] = 1
result = np.append(result, richtig_vec.reshape(len(y_pred),1), axis= 1)

conf_idx = np.where(np.max(pred, axis =1) > c_thresh)[0]
conf_vec = np.zeros(len(y_pred))
conf_vec[conf_idx] = 1
result = np.append(result, conf_vec.reshape(len(y_pred),1), axis= 1)

f_nc = ((conf_vec == 0) & (richtig_vec == 0)).sum()
f_c  = ((conf_vec == 1) & (richtig_vec == 0)).sum()
t_nc = ((conf_vec == 0) & (richtig_vec == 1)).sum()
t_c  = ((conf_vec == 1) & (richtig_vec == 1)).sum()


print(f_nc/(f_c+f_nc))
print(t_c/(f_c+t_c))


#df_result = pd.DataFrame(data=result)
df_result = pd.DataFrame(data=result, columns = ['pred1', 'pred2', 'y1', 'y2', 'acc' , 'conf' ])

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

#%% confident training
N = len(X_trn)
perplexity = 40
#perplexity = (np.log(N))**2

time_start = time.time()
tsne = TSNE (n_components=2, verbose=1, perplexity = perplexity, n_iter=300)
tsne_results = tsne.fit_transform(X_trn)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%
#plt.figure(figsize=(16,10))
plt.figure()
df_subset = pd.DataFrame(data = tsne_results, columns = ["tsne-2d-one","tsne-2d-two"])
df_subset['y'] = y_trn[:,0]
#df_subset['tsne-2d-two'] = tsne_results[:,1]
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", len(np.unique(y_trn))),
    data = df_subset,
    legend="full",
    alpha=0.3
)

#%% 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.tree import plot_tree
plt.close('all')

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(tsne_results, y_trn[:,0])
#clf.fit(X_trn, y_trn[:,0])
y_predicted = clf.predict(tsne_results)
#y_predicted = clf.predict(X_trn)
#score = clf.score(X_test, y_test[:,0])
#print(score)
plt.figure(figsize=(15,10))
a = plot_tree(clf, 
              feature_names = ['f{}'.format(i) for i in range(2)], 
              class_names = ['Normal','AF'], 
              filled=True, 
              rounded=True, 
              fontsize=14)

#%%
from matplotlib.colors import ListedColormap
h = 0.2
X_train = tsne_results.copy()
y_train = y_trn.copy()
x_min, x_max = tsne_results[:, 0].min() - .5, tsne_results[:, 0].max() + .5
y_min, y_max = tsne_results[:, 1].min() - .5, tsne_results[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)

figure = plt.figure(figsize=(27, 9))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#if ds_cnt == 0:
#    ax.set_title("Input data")
# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
    
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
if ds_cnt == 0:
    ax.set_title(name)
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        size=15, horizontalalignment='right')

#%%
# create meshgrid
resolution = 100 # 100x100 background pixels
X2d_xmin, X2d_xmax = np.min(tsne_results[:,0]), np.max(tsne_results[:,0])
X2d_ymin, X2d_ymax = np.min(tsne_results[:,1]), np.max(tsne_results[:,1])
xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
background_model = KNeighborsClassifier(n_neighbors=1).fit(tsne_results, y_predicted) 
voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
voronoiBackground = voronoiBackground.reshape((resolution, resolution))

#plot
plt.contourf(xx, yy, voronoiBackground)
plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y)
plt.show()

#%% SVM different margins
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# figure number
fignum = 1

# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):

    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)
    print(clf.score(X,Y))

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()