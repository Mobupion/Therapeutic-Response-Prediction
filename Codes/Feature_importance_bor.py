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

#GPU selection
gpu_id = "5"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)

#Self-definition
file = "Bortezomib"
drug = "PS341" #PS341/Dex
data_split = "PD-R_response" #PD-R_response/NR-R_response

clu_channels = 9
cv = 5
ran_seed = 0

#Data input
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response+trial+drug.csv".format(file), header = 0, index_col = 0)
dataX = data_T.drop(columns = ["PD-R_response","NR-R_response","trial","drug"])

data_T = data_T[pd.notnull(data_T[data_split])]
train_xy = data_T.loc[[i in [25, 40] for i in data_T["trial"]]]
test_xy = data_T.loc[data_T["trial"] == 39].loc[data_T["drug"] == drug]

trainX = train_xy.drop(columns = ["PD-R_response","NR-R_response","trial","drug"])
testX = test_xy.drop(columns = ["PD-R_response","NR-R_response","trial","drug"])
trainY = train_xy[data_split]
testY = test_xy[data_split]

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

#Model loading
clf = tf.keras.models.load_model("/raid/mobu/3_P_best_model/{}+{}+{}_{}_channels({})_{}-cv_best-model_{}".format(file,drug,data_split,"DR",clu_channels,cv,ran_seed))

auc = roc_auc_score(testY.values, [i[1] for i in clf.predict(testX_mp)])
acc = accuracy_score(testY.values, np.round([i[1] for i in clf.predict(testX_mp)]))
f1 = f1_score(testY.values, np.round([i[1] for i in clf.predict(testX_mp)]))

print("F1: {}\nAccuracy: {}\nROC_AUC: {}".format(f1,acc,auc))

#Pre-Interpretability
clf.X_ = trainX_mp
clf.y_ = trainY_binary
clf._model = clf

#Model explaination by simply-explainer: global, local
simp_explainer = AggMapNet.simply_explainer(clf, mp)
global_simp_importance = simp_explainer.global_explain(trainX_mp, trainY_binary)
#local_simp_importance = simp_explainer.local_explain(trainX_mp[[0]], trainY_binary[[0]])

global_simp_importance.to_csv('/raid/mobu/5_feature_importance/{}_simp-importance_channels({})_{}.csv'.format(file,clu_channels,ran_seed), \
	na_rep='NA')

#Model explaination by shapley-explainer: global, local
shap_explainer = AggMapNet.shapley_explainer(clf, mp)
global_shap_importance = shap_explainer.global_explain(trainX_mp)
local_shap_importance = shap_explainer.local_explain(trainX_mp[[0]])

global_shap_importance.to_csv('/raid/mobu/5_feature_importance/{}_shap-importance_channels({})_{}.csv'.format(file,clu_channels,ran_seed), \
	na_rep='NA')
