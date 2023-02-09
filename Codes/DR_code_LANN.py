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
file = "LNN-BRCA"
#Bortezomib/Bortezomib_PD-R/Cisplatin/Paclitaxel/CN-AML/MBC/KT-1/KT-2/Docetaxel/Infliximab-1/Infliximab-2
#NPC/BRCA_Cisplatin/PARPI/KT-3/RA-PBMC/RA-CD4/RA-CD14
clu_channels = 7
cv = 5
ran_seed = 0

#Data input
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-ER_Status+time+relapse+response.csv".format(file), header = 0, index_col = 0)

#data_normal = pd.read_csv("/raid/saiki/1_new-data/1_DR_gene-list.csv", header = 0, index_col = 0)

'''
dataX = data_T.drop(columns = "response")
a = [x.upper() for x in dataX.columns]
a = [x.replace("-","") for x in a]
dataX.columns = a
dataX = dataX.loc[:,~dataX.columns.duplicated(keep = False)]
dataX = dataX.reindex(columns = data_normal["Gene"])
dataX = dataX.dropna(axis = 1, how = "all")
'''

#Pre-processing
data_T = data_T.loc[pd.notnull(data_T["response"])]
dataX = data_T.drop(columns = ["ER_Status","time_to_relapse","relapse_(1=True)","response"])
dataY = data_T["response"]

#Training-test set splitting
import random
tt_dic = {}

for i, j in data_T.groupby(by = ["ER_Status"], axis = 0):
	random.seed(ran_seed)
	j_l = list(j.index)
	if i == "ER+":
		trnum = 75
	else:
		trnum = 30
	tr_ones = random.sample(j_l, trnum)  #random sampling
	test_ones = j.index[[x not in tr_ones for x in j.index]]
	tt_dic["train_set"] = tt_dic.get("train_set",[]) + tr_ones
	tt_dic["test_set"] = tt_dic.get("test_set",[]) + list(test_ones)

#f_tt = open("/raid/saiki/TT-split/0_Final-Patient_TT-sample-id.txt", "a+")
#f_tt.write("Dataset: {}\nRandom state: {}\nTrain set({}): {}\nTest set({}): {}\n\n".format(file, ran_seed, len(tt_dic["train_set"]), tt_dic["train_set"], len(tt_dic["test_set"]), tt_dic["test_set"]))
#f_tt.close()
#print("Train set: {}\nTest set: {}".format(len(tt_dic["train_set"]), len(tt_dic["test_set"])))

trainX, trainY = dataX.loc[tt_dic["train_set"]], dataY.loc[tt_dic["train_set"]]
testX, testY = dataX.loc[tt_dic["test_set"]], dataY.loc[tt_dic["test_set"]]

#AggMap generation
if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed)):
	mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))
else:
	if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed)):
		mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed))
	else:
		mp = AggMap(dataX, metric = "correlation")
	mp.fit(cluster_channels = clu_channels)
	mp.save("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))

#AggMap transformation
trainX_mp = mp.batch_transform(trainX.values)
testX_mp = mp.batch_transform(testX.values)
trainY_binary = tf.keras.utils.to_categorical(trainY.values,2)
testY_binary = tf.keras.utils.to_categorical(testY.values,2)

#Cross-validation + randomgrid search
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=ran_seed)
scoring = {"AUC": "roc_auc"}

clf = AggMapNet.MultiClassEstimator(gpuid = gpu_id)
dense_layer_list = [[64],[128],[256]]

random_grid = {"n_inception": [1,2],"conv1_kernel_size": list(range(11,25,2)), "dense_layers": dense_layer_list, "dropout": [0.0,0.1,0.3,0.5], \
"epochs": [50,100,200], "patience": [30], "batch_size": [64,128,256], "lr": [1e-4, 3e-4, 5e-4], \
"batch_norm": [True, False],"random_state": [32]}

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
cv_results.to_csv('/raid/mobu/2_cv-results/{}_{}_channels({})_{}-cv-results_{}.csv'.format(file,"DR",clu_channels,cv,ran_seed), \
	na_rep='NA')
clf._model.save("/raid/mobu/3_P_best_model/{}_{}_channels({})_{}-cv_best-model_{}".format(file,"DR",clu_channels,cv,ran_seed))
f_tt = open("/raid/mobu/1_DR_{}-cv-results_{}.txt".format(cv,ran_seed), "a+")
f_tt.write("Dataset: {}\nChannels: {}\nBest parameters: {}\nF1: {}\nAccuracy: {}\nROC_AUC: {}\n\n".format(file, clu_channels, parameters, f1, acc, auc))
f_tt.close()
