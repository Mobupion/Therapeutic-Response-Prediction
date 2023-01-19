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

#选择哪个GPU进行计算，“2”代表编号为2的GPU
gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)

#Self-definition
file = "CN-AML"
#Bortezomib/Bortezomib_PD-R/Cisplatin/Paclitaxel/CN-AML/MBC/KT-1/KT-2/Docetaxel/Infliximab-1/Infliximab-2
#NPC/BRCA_Cisplatin/PARPI/KT-3/RA-PBMC/RA-CD4/RA-CD14
clu_channels = 7
ran_seed = 0

#读入数据集
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response.csv".format(file), header = 0, index_col = 0)

#数据集的预处理，提取出特征集和标签集
dataX = data_T.drop(columns = "response")
dataY = data_T["response"]

#创建AggMap对象
if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed)):
	mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))
else:
	if os.path.isfile("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed)):
		mp = loadmap("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed))
	else:
		mp = AggMap(dataX, metric = "correlation")
	mp.fit(cluster_channels = clu_channels)
	mp.save("/raid/mobu/1_aggmap/{}_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))

#params_list
params_list = [\
{"conv1_kernel_size": 9, "dense_layers": [256],  "epochs": 150, "patience": 30, "batch_size": 128, "lr": 1.5e-4, "random_state": 32, "verbose": 0}, \
{"conv1_kernel_size": 11, "dense_layers": [128],  "epochs": 200, "patience": 30, "batch_size": 128, "lr": 1e-4, "random_state": 32, "verbose": 0}, \
{"conv1_kernel_size": 13, "dense_layers": [256],  "epochs": 150, "patience": 30, "batch_size": 128, "lr": 1.5e-4, "random_state": 32, "verbose": 0}, \
{"conv1_kernel_size": 15, "dense_layers": [128],  "epochs": 200, "patience": 30, "batch_size": 128, "lr": 1e-4, "random_state": 32, "verbose": 0}, \
{"conv1_kernel_size": 13, "dense_layers": [128,64],  "epochs": 150, "patience": 30, "batch_size": 128, "lr": 1.5e-4, "random_state": 32, "verbose": 0}, \
]

#Cross-validation
i = 4
parameters = params_list[i]
parameters["gpuid"] = gpu_id
dataX_mp = mp.batch_transform(dataX.values)
dataY_binary = tf.keras.utils.to_categorical(dataY.values,2)

all_result = pd.DataFrame()
loo = LeaveOneOut()
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

all_result.to_csv('/raid/mobu/4_Results/{}_channels({})_loocv_All-results_{}_{}.csv'.format(file,clu_channels,ran_seed,i), \
	na_rep='NA')
f_tt = open("/raid/mobu/1_DR_loocv-results_{}.txt".format(ran_seed), "a+")
f_tt.write("Dataset: {}\nChannels: {}\nParameters: {}\nF1: {}\nAccuracy: {}\nROC_AUC: {}\n\n".format(file, clu_channels, parameters, f1, acc, auc))
f_tt.close()
