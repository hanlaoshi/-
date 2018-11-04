#!---* coding: utf-8 --*--
#!/usr/bin/python
"""
ProjectName:baseline_NLP
@Author:Aifu Han
Date:2018.8.14
"""

import pandas as pd, numpy as np
import xgboost as xgb
# from xgboost import XGBClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
t1=time.time()
print('Reading...and  the timer is starting')
train = pd.read_csv('./train_set.csv')
#
test = pd.read_csv('./test_set.csv')
# column = "id"
test_id = pd.read_csv('./test_set.csv',)[["id"]].copy()
# test_id = test[column]
print('reading over,next is for loop...')
column1 ="word_seg"
# column2 = "article"
# column3 = "text"
n = train.shape[0]

print('the for loop is over,and next is training model')

vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column1])
test_term_doc = vec.transform(test[column1])

y=(train["class"]-1).astype(int)

print('training the model by XGBOOST')

#开始用xgboost训练模型
dtrain = xgb.DMatrix(trn_term_doc, label=y)
dtest = xgb.DMatrix(test_term_doc)  # label可以不要，此处需要是为了测试效果
# param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':19}  # 参数
# watchlist  = [(dtrain,'train'),(dtest,'test')]  # 这步可以不要，用于测试效果
# num_round = 100  # 循环次数
# clf = xgb.train(param, dtrain, num_round, watchlist)
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eval_metic']='mlogloss'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 19

watchlist = [(dtrain, '训练误差')]
num_round = 300

clf = xgb.train(param, dtrain, num_round, evals=watchlist)


# model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
# clf = LogisticRegression(C=4, dual=True)
# clf.fit(trn_term_doc, y)
preds=clf.predict(dtest)
# print(preds)
#保存概率文件
test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('./8.16_lr_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('./sub_lr_baseline.csv',index=None)
t2=time.time()
print("the project is done and time use:",t2-t1)
