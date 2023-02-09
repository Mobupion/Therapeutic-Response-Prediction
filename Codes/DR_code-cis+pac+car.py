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
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold

import seaborn as sns
import matplotlib.pyplot as plt
from aggmap import AggMap, AggMapNet, show, loadmap

class BBOX:
	"""docstring for ClassName"""
	def __init__(self, file, clu_channels, cv, ran_seed, gpu_id):
		
		self.file = file
		self.clu_channels = clu_channels
		self.cv = cv
		self.ran_seed = ran_seed
		self.gpu_id = gpu_id
		
	def train_test_process(self, all_result, trainX_mp, testX_mp, trainY_binary, testY_binary, i):

		if i == 1:
			#Cross-validation + randomgrid search
			skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.ran_seed)
			scoring = {"AUC": "roc_auc"}

			clf = AggMapNet.MultiClassEstimator(gpuid = self.gpu_id)
			dense_layer_list = [[64],[128],[256]]

			random_grid = {"n_inception": [1,2],"conv1_kernel_size": list(range(11,25,2)), "dense_layers": dense_layer_list, "dropout": [0.0,0.1,0.3,0.5], \
			"epochs": [50,100,200], "patience": [30], "batch_size": [64,128,256], "lr": [1e-4, 3e-4, 5e-4], \
			"batch_norm": [True, False],"random_state": [32]}

			clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 30, \
				scoring = scoring, n_jobs = None, cv = skf.split(trainX_mp, trainY.values), refit = "AUC", verbose = 0, random_state = self.ran_seed)

			clf_random.fit(trainX_mp, trainY_binary)
			best_estimator = clf_random.best_estimator_
			self.parameters = clf_random.best_params_
			#cv_results = pd.DataFrame(clf_random.cv_results_)
			f_tt = open("/raid/mobu/1_DR-{}_cv-results_channels({})_{}.txt".format(self.file,self.clu_channels,self.ran_seed), "a+")
			f_tt.write("Best parameters: {}\n\n".format(self.parameters))
			f_tt.close()

			clf = best_estimator
			self.parameters["gpuid"] = self.gpu_id
		else:
			clf = AggMapNet.MultiClassEstimator(**self.parameters)
			clf.fit(trainX_mp, trainY_binary)

		auc = roc_auc_score(testY.values, [i[1] for i in clf.predict_proba(testX_mp)])
		acc = accuracy_score(testY.values, clf.predict(testX_mp))
		f1 = f1_score(testY.values, clf.predict(testX_mp))

		all_result["split-{}".format(i)] = [f1, acc, auc]
		self.all_result = all_result
		#Performace record
		#cv_results.to_csv('/raid/mobu/2_cv-results/{}_{}_channels({})_{}-cv-results_{}.csv'.format(file,"DR",clu_channels,cv,ran_seed), \
		#	na_rep='NA')
		#clf._model.save("/raid/mobu/3_P_best_model/{}_{}_channels({})_{}-cv_best-model_{}".format(file,"DR",clu_channels,cv,ran_seed))
		f_tt = open("/raid/mobu/1_DR-{}_cv-results_channels({})_{}.txt".format(self.file,self.clu_channels,self.ran_seed), "a+")
		f_tt.write("n_fold: {}\nF1: {}\nAccuracy: {}\nROC_AUC: {}\n\n".format(i, f1, acc, auc))
		f_tt.close()

#GPU selection
gpu_id = "7"
#os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
#physical_gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_gpus[0], True)

#Self-definition
file = "Cisplatin" #Carboplatin/Cisplatin/Paclitaxel
cancer = "BLCA" # (UCEC)/pancancer # BLCA/CESC/LUAD/pancancer # (BRCA)
clu_channels = 5
cv = 5
ran_seed = 0

#Data input
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response+cancer.csv".format(file), header = 0, index_col = 0)
cv_split = pd.read_csv("/raid/mobu/0_datasets/{}_phenotype.csv".format(file), header = 0, index_col = 0)

dataX = data_T.drop(columns = ["response","cancer"])

#AggMap generation
if os.path.isfile("/raid/mobu/1_aggmap/{}-all_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed)):
	mp = loadmap("/raid/mobu/1_aggmap/{}-all_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))
else:
	if os.path.isfile("/raid/mobu/1_aggmap/{}-all_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed)):
		mp = loadmap("/raid/mobu/1_aggmap/{}-all_DR_channels({})_{}-cv_{}.mp".format(file,5,5,ran_seed))
	else:
		mp = AggMap(dataX, metric = "correlation")
	mp.fit(cluster_channels = clu_channels)
	mp.save("/raid/mobu/1_aggmap/{}-all_DR_channels({})_{}-cv_{}.mp".format(file,clu_channels,5,ran_seed))

if cancer == "pancancer":
	data_T = data_T.drop(columns = "cancer")
else:
	data_T = data_T[data_T["cancer"]==cancer].drop(columns = "cancer")

file = "{}+{}".format(file,cancer)
process_box = BBOX(file, clu_channels, cv, ran_seed, gpu_id)
all_result = pd.DataFrame()
for i in cv_split.columns:
	trainXY_index = cv_split[cv_split.loc[:,i]=="train"].index
	testXY_index = cv_split[cv_split.loc[:,i]=="validation"].index

	trainX = data_T.loc[[i in trainXY_index for i in data_T.index]].drop(columns = "response")
	testX = data_T.loc[[i in testXY_index for i in data_T.index]].drop(columns = "response")
	trainY = data_T.loc[[i in trainXY_index for i in data_T.index], "response"]
	testY = data_T.loc[[i in testXY_index for i in data_T.index], "response"]

	#AggMap transformation
	trainX_mp = mp.batch_transform(trainX.values)
	testX_mp = mp.batch_transform(testX.values)
	trainY_binary = tf.keras.utils.to_categorical(trainY.values,2)
	testY_binary = tf.keras.utils.to_categorical(testY.values,2)

	i = eval(i)-2
	process_box.train_test_process(all_result, trainX_mp, testX_mp, trainY_binary, testY_binary, i)

process_box.all_result.index = ["Test_F1", "Test_Accuracy", "Test_AUC"]
process_box.all_result["Mean"] = all_result.mean(axis = 1)

process_box.all_result.to_csv('/raid/mobu/4_Results/{}_{}_channels({})_{}-cv_All-results_{}.csv'.format(file,"DR",clu_channels,cv,ran_seed), \
	na_rep='NA')
