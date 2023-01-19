import warnings
warnings.filterwarnings("ignore")
import os.path
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, LeaveOneOut

import seaborn as sns
import matplotlib.pyplot as plt
from aggmap import AggMap, AggMapNet, show, loadmap

#Self-definition
file = "Infliximab-1"
file_2 = "Infliximab-2"
#KT-1/KT-2/Infliximab-1/Infliximab-2

clu_channels = 5
cv = 5
ran_seed = 0

#读入数据集
data_T = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response.csv".format(file), header = 0, index_col = 0)
data_T_2 = pd.read_csv("/raid/mobu/0_datasets/{}_log2expression-response.csv".format(file_2), header = 0, index_col = 0)

dataX = data_T.drop(columns = "response")
dataX_2 = data_T_2.drop(columns = "response")

dataX = pd.concat([dataX,dataX_2],axis=0)
dataX_new = StandardScaler().fit_transform(dataX)
dataX = pd.DataFrame(dataX_new, index = dataX.index, columns = dataX.columns)

#创建AggMap对象
if os.path.isfile("/raid/mobu/1_aggmap/{}_DR-Z_channels({})_{}-cv_{}.mp".format(file[:-2],clu_channels,5,ran_seed)):
    mp = loadmap("/raid/mobu/1_aggmap/{}_DR-Z_channels({})_{}-cv_{}.mp".format(file[:-2],clu_channels,5,ran_seed))
else:
    if os.path.isfile("/raid/mobu/1_aggmap/{}_DR-Z_channels({})_{}-cv_{}.mp".format(file[:-2],5,5,ran_seed)):
        mp = loadmap("/raid/mobu/1_aggmap/{}_DR-Z_channels({})_{}-cv_{}.mp".format(file[:-2],5,5,ran_seed))
    else:
        mp = AggMap(dataX, metric = "correlation")
    mp.fit(cluster_channels = clu_channels)
    mp.save("/raid/mobu/1_aggmap/{}_DR-Z_channels({})_{}-cv_{}.mp".format(file[:-2],clu_channels,5,ran_seed))
