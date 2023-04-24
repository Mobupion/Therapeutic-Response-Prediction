import warnings
warnings.filterwarnings("ignore")
import os.path
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, LeaveOneOut

import seaborn as sns
import matplotlib.pyplot as plt
from aggmap import AggMap, AggMapNet, show, loadmap

#GPU selection
gpu_id = "2"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)

#Self-definition
file = "KT-2"
file_2 = False
#KT-1/KT-2/Infliximab-1/Infliximab-2

clu_channels = 7
cv = 5
ran_seed = 42

#Data Input
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response.csv".format(file), header = 0, index_col = 0)

#Pre-processing
dataX = data_T.drop(columns = "response")
dataY = data_T["response"]

#Train-test split
import random
tr_size = 1 - (1/cv)
tt_dic = {}

for i, j in data_T.groupby(by = ["response"], axis = 0):
	random.seed(ran_seed)
	j_l = list(j.index)
	tr_ones = random.sample(j_l, round(tr_size*len(j_l)))  #random sampling
	test_ones = j.index[[x not in tr_ones for x in j.index]]
	tt_dic["train_set"] = tt_dic.get("train_set",[]) + tr_ones
	tt_dic["test_set"] = tt_dic.get("test_set",[]) + list(test_ones)

#f_tt = open("/raid/saiki/TT-split/0_Final-Patient_TT-sample-id.txt", "a+")
#f_tt.write("Dataset: {}\nRandom state: {}\nTrain set({}): {}\nTest set({}): {}\n\n".format(file, ran_seed, len(tt_dic["train_set"]), tt_dic["train_set"], len(tt_dic["test_set"]), tt_dic["test_set"]))
#f_tt.close()
#print("Train set: {}\nTest set: {}".format(len(tt_dic["train_set"]), len(tt_dic["test_set"])))

trainX, trainY = dataX.loc[tt_dic["train_set"]], dataY.loc[tt_dic["train_set"]]
testX, testY = dataX.loc[tt_dic["test_set"]], dataY.loc[tt_dic["test_set"]]

if file_2:
	data_T_2 = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response.csv".format(file_2), header = 0, index_col = 0)
	trainX = data_T.drop(columns = "response")
	testX = data_T_2.drop(columns = "response")
	trainY = data_T["response"]
	testY = data_T_2["response"]

#AggMap generation
if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_0.mp".format(file[:-2],clu_channels,5)):
    mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_0.mp".format(file[:-2],clu_channels,5))
else:
    if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_0.mp".format(file[:-2],5,5)):
        mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_0.mp".format(file[:-2],5,5))
    else:
        print("Error!\n")
    mp.fit(cluster_channels = clu_channels)
    mp.save("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_0.mp".format(file[:-2],clu_channels,5))

#AggMap transformation
trainX_mp = mp.batch_transform(trainX.values, scale_method = 'standard')
testX_mp = mp.batch_transform(testX.values, scale_method = 'standard')
trainY_binary = tf.keras.utils.to_categorical(trainY.values,2)
testY_binary = tf.keras.utils.to_categorical(testY.values,2)

#Cross-validation + randomgrid search
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=ran_seed)
scoring = {"AUC": "roc_auc"}

clf = AggMapNet.MultiClassEstimator(gpuid = gpu_id)
dense_layer_list = [[64, 32],[128, 32],[256, 32]]

random_grid = {"n_inception": [1,2],"conv1_kernel_size": [3,5,7,9,11,13,15], "dense_layers": dense_layer_list, \
"epochs": [50,100,200], "patience": [60], "batch_size": [8, 16, 32, 64], "lr": [1e-4, 3e-4, 5e-4], "random_state": [32]}

clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 30, \
	scoring = scoring, n_jobs = None, cv = skf.split(trainX_mp, trainY.values), refit = "AUC", verbose = 0, random_state = ran_seed)

clf_random.fit(trainX_mp, trainY_binary)
best_estimator = clf_random.best_estimator_
parameters = clf_random.best_params_
cv_results = pd.DataFrame(clf_random.cv_results_)

#Evaluation
clf = best_estimator

auc = roc_auc_score(testY.values, [i[1] for i in clf.predict_proba(testX_mp)])
acc = accuracy_score(testY.values, clf.predict(testX_mp))
f1 = f1_score(testY.values, clf.predict(testX_mp))

#print("F1: {}\nAccuracy: {}\nROC_AUC: {}".format(f1, acc, auc))

#Performace record
if file_2:
	file = "{}(train)-{}(test)".format(file, file_2)
cv_results.to_csv('/raid/mobu/2_cv-results/{}_{}_channels({})_{}-cv-results_{}.csv'.format(file,"NEW",clu_channels,cv,ran_seed), \
	na_rep='NA')
clf._model.save("/raid/mobu/3_P_best_model/{}_{}_channels({})_{}-cv_best-model_{}".format(file,"NEW",clu_channels,cv,ran_seed))
f_tt = open("/raid/mobu/1_NEW_{}-cv-results_{}.txt".format(cv,ran_seed), "a+")
f_tt.write("Dataset: {}\nChannels: {}\nBest parameters: {}\nF1: {}\nAccuracy: {}\nROC_AUC: {}\n\n".format(file, clu_channels, parameters, f1, acc, auc))
f_tt.close()

#leave-one-out CV
if file_2:
	pass
else:
	parameters["gpuid"] = gpu_id
	all_result = pd.DataFrame()

	loo = LeaveOneOut()
	dataX_mp = mp.batch_transform(dataX.values)
	dataY_binary = tf.keras.utils.to_categorical(dataY.values,2)

	for train_index, test_index in loo.split(dataX_mp, dataY.values):
		trainX_mp, trainY_binary = dataX_mp[train_index], dataY_binary[train_index]
		testX_mp, testY_binary = dataX_mp[test_index], dataY_binary[test_index]
	
		clf = AggMapNet.MultiClassEstimator(**parameters)
		clf.fit(trainX_mp, trainY_binary)

		#Evaluation
		p_prob = clf.predict_proba(testX_mp)
		p_label = clf.predict(testX_mp)

		all_result["sample-{}".format(test_index[0])] = [p_prob[0][0], p_prob[0][1], p_label[0], dataY.values[test_index[0]]]

	all_result.index = ["Predicted_proba_0", "Predicted_proba_1", "Predicted_label", "True_label"]
	
	auc = roc_auc_score(all_result.loc["True_label"].values, all_result.loc["Predicted_proba_1"].values)
	acc = accuracy_score(all_result.loc["True_label"].values, all_result.loc["Predicted_label"].values)
	f1 = f1_score(all_result.loc["True_label"].values, all_result.loc["Predicted_label"].values)

	all_result.to_csv('/raid/mobu/4_Results/{}_channels({})_loocv_NEW-results_{}.csv'.format(file,clu_channels,ran_seed), \
		na_rep='NA')
	f_tt = open("/raid/mobu/1_NEW_loocv-results_{}.txt".format(ran_seed), "a+")
	f_tt.write("Dataset: {}\nChannels: {}\nParameters: {}\nF1: {}\nAccuracy: {}\nROC_AUC: {}\n\n".format(file, clu_channels, parameters, f1, acc, auc))
	f_tt.close()
