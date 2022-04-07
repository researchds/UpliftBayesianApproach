import sys
from sklift.datasets import fetch_lenta
from sklift.models import ClassTransformation
from sklift.metrics import uplift_at_k
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklift.models import TwoModels
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
from causalml.feature_selection.filters import FilterSelect
from UMODL_featureImportance import getImportantVariables_UMODL_ForMultiProcessing
from sklearn.model_selection import StratifiedKFold
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklift.models import ClassTransformation
import multiprocessing as mp
import pickle
import os
import time
from causalml.metrics import plot_gain, auuc_score,plot_qini,qini_score
import math
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor, BaseXLearner, BaseDRLearner
from random import randrange, randint
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)
import random
random.seed(10)
np.random.seed(10)
# %matplotlib inline




_nb_counter = []
_start_counter = []
_start_time_counter = []
_deltatime_counter = []
_NumberOfCounters=14
for i in range(0,_NumberOfCounters):
    _nb_counter.append(0)
    _start_counter.append(False)
    _start_time_counter.append(time.time())
    _deltatime_counter.append(0)

def start_counter(i):
    _nb_counter[i]=_nb_counter[i]+1
    _start_counter[i]=True
    _start_time_counter[i]=time.time()  
    
def stop_counter(i):
    _start_counter[i]=False
    diff = time.time()  - _start_time_counter[i]
    _deltatime_counter[i] = _deltatime_counter[i]+diff
    _start_time_counter[i]=time.time()  



def LearnInParallelWithDifferentUnnecessaryCols(args):
    global experiment_group_column
    global y_name
    global ExtraColsVsUMODL_ListOfFeatures
    global ExtraColsVsMethodsAndListOfFeatures
    global foldNum
    
    lentaTrain=args[0].copy()
    lentaTest=args[1].copy()
    extraCol=args[2]
    
    
    methods=['F','LR','KL','Chi','ED','UMODL','ALLOriginal']
    lentaTest[experiment_group_column]=lentaTest[experiment_group_column].astype(int)
    lentaTrain[experiment_group_column]=lentaTrain[experiment_group_column].astype(int)
    lentaTrain[y_name]=lentaTrain[y_name].astype(float)
    lentaTest[y_name]=lentaTest[y_name].astype(float)
    
    lentaTrain=lentaTrain.reset_index(drop=True)
    lentaTest=lentaTest.reset_index(drop=True)

    Results={}
    for method in methods:
#         print("ExtraCOL ",extraCol,"Method",method)
#         print("Method ",method)

        if method=="UMODL":
            cols=list(ExtraColsVsUMODL_ListOfFeatures[str(extraCol)]).copy()
        elif method=='ALLOriginal':
            cols=list(X_names_original)
        else:
            cols=ExtraColsVsMethodsAndListOfFeatures[str(extraCol)][method].copy()
        cols.sort()
        print("Method",method," Extra columns number is ",extraCol," Columns are ",cols)
#         print("Columns are ",cols)
        #2Model
        lentaTest[experiment_group_column]=lentaTest[experiment_group_column].astype(int)
        lentaTrain[experiment_group_column]=lentaTrain[experiment_group_column].astype(int)
        lentaTrain[y_name]=lentaTrain[y_name].astype(float)
        lentaTest[y_name]=lentaTest[y_name].astype(float)
        


#         estimator_trmnt=RandomForestClassifier(random_state=55)#,max_features=3,max_depth=10,min_samples_leaf=100)
#         estimator_ctrl=RandomForestClassifier(random_state=55)#,max_features=3,max_depth=10,min_samples_leaf=100)
        estimator_trmnt=LogisticRegression(random_state=60)#,max_features=3,max_depth=10,min_samples_leaf=100)
        estimator_ctrl=LogisticRegression(random_state=60)#,max_features=3,max_depth=10,min_samples_leaf=100)
        
        print("lentaTrain[lentaTrain[experiment_group_column]==1][y_name].reset_index() ",lentaTrain[lentaTrain[experiment_group_column]==1][y_name].reset_index())
        estimator_trmnt.fit(lentaTrain[lentaTrain[experiment_group_column]==1][cols].values, lentaTrain[lentaTrain[experiment_group_column]==1][y_name].values)
        trmntPreds=estimator_trmnt.predict_proba(lentaTest[cols].values)

        estimator_ctrl.fit(lentaTrain[lentaTrain[experiment_group_column]==0][cols].values, lentaTrain[lentaTrain[experiment_group_column]==0][y_name].values)
        CtrlPreds=estimator_ctrl.predict_proba(lentaTest[cols].values)
        
        treatmentPreds=[]
        for index in range(len(trmntPreds)):
            treatmentPreds.append(trmntPreds[index][1])
            
        ControlPreds=[]
        for index in range(len(CtrlPreds)):
            ControlPreds.append(CtrlPreds[index][1])
            
        preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]
        
#         predsSeries = pd.DataFrame(preds,columns=['preds'])
        
        df_preds=pd.DataFrame()
        df_preds['YProbInTrt']=treatmentPreds
        df_preds['YProbInCtrl']=ControlPreds
        df_preds['Predictions']=preds
        # df_preds['Predictions']=predsNoisty
        df_preds['Treatment']=lentaTest[experiment_group_column].reset_index()[experiment_group_column].astype(int)
        df_preds['y']=lentaTest[y_name].reset_index()[y_name].astype(int)
        df_preds.to_csv("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/predictions/PredictionsOfEachIndividual_ExtraCols_"+str(extraCol)+"_Method_"+method+"_"+str(foldNum)+".csv")
#         upliftModel= TwoModels(estimator_trmnt = estimator_trmnt, estimator_ctrl = estimator_ctrl,method='vanilla')
        
        
        print("Method",method," Extra columns number is ",extraCol," Columns are ",cols," and lentatrain cols is ",lentaTrain[cols].values)
        
#         upliftModel.fit(X = lentaTrain[cols].values, 
#                          treatment = lentaTrain[experiment_group_column].values,
#                          y = lentaTrain[y_name].values)
# #         df_preds=pd.DataFrame.from_dict(upliftModel.predict(lentaTest[cols].values))
#         df_preds=pd.DataFrame()
#         df_preds['Predictions']=upliftModel.predict(lentaTest[cols].values)
#         df_preds['Treatment']=lentaTest[experiment_group_column].reset_index()[experiment_group_column].astype(int)
#         df_preds['y']=lentaTest[y_name].reset_index()[y_name].astype(int)
        
        res=qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment'])
        
        print("Will save this in results dictionary : ","2M"+str(extraCol)+method)
        Results["2M"+str(extraCol)+method]=res     
        
    return Results


start_counter(1)
old_stdout = sys.stdout
# log_file = open("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/CriteoRandIntLR__FINALRESULT_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE.log","w")
log_file = open("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/CriteoRandIntLR__FINALRESULT_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE.log","w")
sys.stdout = log_file


df = pd.read_csv("~/../../data/userstorage/mXXXXXXXX/Datasets/criteo-uplift-v2.1.csv")

# df = pd.read_csv("~/../../data/userstorage/mXXXXXXXX/Datasets/megafon_dataset.csv")

# df=df.sample(frac=1)
df = df.sample(10000,random_state=80)

df.drop(['exposure','conversion'],axis=1,inplace=True)

y_name='visit'
experiment_group_column='treatment'

# y_name='conversion'
# experiment_group_column='treatment_group'
OriginalColumnsCount=df.shape[1]-2
NumberOfNoisyDuplicates=16
Total_NumberOfNoisyCols=100
TotalNumberOfNoisy=NumberOfNoisyDuplicates*OriginalColumnsCount
nFolds=10
# UnnecessaryCols=[100,200,300,400,500,600,700,800,900,1000]
# UnnecessaryCols=[22,44,66,88,110,132,154,176,192]
# UnnecessaryCols=[0,5,10,15,20,25,30]
# UnnecessaryCols=[0]
# UnnecessaryCols=[0,10,20,30,40,50,60,70,80,90,100]
UnnecessaryCols=[0,10,20,30,40,50,60,70,80,90,100]
# df[experiment_group_column].replace({'control':0,'treatment':1},inplace=True)
df = df[[c for c in df if c not in [experiment_group_column, y_name]] + [experiment_group_column, y_name]]

X_names=df.columns[:-2]
X_names_original=df.columns[:-2]

  
df = df[[experiment_group_column, y_name]+[c for c in df if c not in [experiment_group_column, y_name]]]
X_names=df.columns[2:]

df.to_csv("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/Criteo200NoiseCols.csv")
print("Normally it should be saved")
print("df head")
print(df.head())
print("shape of df ",df.shape)
print("X_names are ",X_names)
#---------------------------------------------------------------------------------

skf = StratifiedKFold(n_splits=nFolds,random_state=60)
# kf = KFold(n_splits=10,random_state=55)

s = skf.split(df,df[y_name])
# s = kf.split(df)
foldNum=0
FINALRESULTS=[]

trainTestIndices=[]
#Generate random variables for each fold
for i in range(Total_NumberOfNoisyCols):
    df['NoisyCol'+str(i)]=-10000
for train_index, test_index in s:
    trainTestIndices.append([train_index,test_index])
    df_t1=df.iloc[test_index][df.iloc[test_index][experiment_group_column]==1]
#     df_t1=df.iloc[test_index].index[df.iloc[test_index][df.iloc[test_index][experiment_group_column]==1]]
    print("df_t1 is ",df_t1)
    TreatmentIndices=df_t1.index.tolist()
    df_t0=df.iloc[test_index][df.iloc[test_index][experiment_group_column]==0]
    ControlIndices=df_t0.index.tolist()
    print("df before adding noise to some indices is ",df)
    print("type of treatment indices list is ",type(TreatmentIndices))
    for i in range(Total_NumberOfNoisyCols):
        df.loc[TreatmentIndices,'NoisyCol'+str(i)]=np.random.normal(0,1, len(TreatmentIndices))
        df.loc[ControlIndices,'NoisyCol'+str(i)]=np.random.normal(0,1, len(ControlIndices))
print("S is ",trainTestIndices)
print('df is ',df.head())
for train_index, test_index in trainTestIndices:
    foldNum+=1
    print("fold num is ",foldNum,"\n\n")
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    print("df_train\n")
    print(df_train)
    print("df_test")
    print(df_test)
    print("df_train shape ",df_train.shape)
    print("df_test shape ",df_test.shape)

    df_train[experiment_group_column]=df_train[experiment_group_column].astype(int)
    df_train[y_name]=df_train[y_name].astype(int)
    
    df_train.to_csv("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/df_train"+str(foldNum)+".csv")
    df_test.to_csv("/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/df_test"+str(foldNum)+".csv")
    
    X_names=df_train.columns[2:]
    print("X_names are ",X_names)
    
    #Feature Selection with UMODL
    pool = mp.Pool(processes=30)
    argumentsToPass=[]
    for colName in X_names:
        l=[df_train,colName,y_name,experiment_group_column]
        argumentsToPass.append(l)

    loaded_list = pool.map(getImportantVariables_UMODL_ForMultiProcessing, argumentsToPass)
    pool.close()
    print("finished UMODL FOR ALL COLUMNS")
    print("LOADED_List before parsing it :\n",loaded_list)
    NewList=[]
    for d in loaded_list: #result is a list of lists. Each element represents a columns
        l={}
        for k in d:
            if len(d[k][1])==1:#If There is only one interval in the data then val is zero
                l[k]=0
            else:
                l[k]=d[k][0][1]#returns the value of the euclidean distance in the founded intervals
        NewList.append(l)
    loaded_list=NewList.copy()
    result = {}
    for d in loaded_list:
        result.update(d)
    result={k: v for k, v in sorted(result.items(), key=lambda item: item[1],reverse=True)}
    print("Result should contain each variable vs its umodl criterion value:\n",result)

    file_name = "/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/MODL_ScoresCriteoRandIntLR_trial2LR_NormalAndUniformRandomVariables192_"+str(foldNum)+".pkl"
    
    open_file = open(file_name, "wb")
    pickle.dump(result, open_file)
    open_file.close()

    
    NumberOfInformativeVariableByExtraColumns={}
    
    ExtraColsVsUMODL_ListOfFeatures={}
    for extraColumn in UnnecessaryCols:
        res = dict((k, result[k]) for k in X_names[:OriginalColumnsCount+extraColumn] if k in result) #get only the columns within the original and extra columns
        print("Res is ",res)
        res={k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True) if v!=0} #sort them by value and get only the infromative variables
        ExtraColsVsUMODL_ListOfFeatures[str(extraColumn)]=res.keys()
        NumberOfInformativeVariableByExtraColumns[extraColumn]=len(ExtraColsVsUMODL_ListOfFeatures[str(extraColumn)])#the number of informative columns
    #Feature Selection with other methods
    print("ExtraColsVsUMODL_ListOfFeatures should contain for each extracolums value, all the non-zero columns included in the extra columns:\n",ExtraColsVsUMODL_ListOfFeatures)
    print("Length is in the following dict\n",NumberOfInformativeVariableByExtraColumns)
    ############################################################################################################################################################
    #State Of art Feature Selection  Methods
    ExtraColsVsMethodsAndListOfFeatures={}
    for extraCols in UnnecessaryCols:
        methods=['F','LR','KL','Chi','ED']
        MethodVsListOfFeatures={}
        df_train[experiment_group_column]=df_train[experiment_group_column].astype(str)
        for method in methods:
            filter_f=FilterSelect()
            print("extraCols is ",extraCols)
            featureXnames=list(X_names[:OriginalColumnsCount+extraCols])
            DataSetToBeUsed=df_train.iloc[:,:OriginalColumnsCount+2+extraCols]
            print("featureXnames ",featureXnames)
            print("DataSetToBeUsed ",DataSetToBeUsed)
            ImportantFeatures = filter_f.get_importance(DataSetToBeUsed,featureXnames, y_name, method, experiment_group_column=experiment_group_column,
                              treatment_group = '1',control_group='0',n_bins=10)

            print("Number of Extracols is ",extraCols)
            print("ImportantFeaturesDataFrame Method ",method," ImportantFeatures are ",ImportantFeatures['feature'].reset_index()['feature'].tolist())
            MethodVsListOfFeatures[method]=ImportantFeatures['feature'].reset_index()['feature'].tolist()[:NumberOfInformativeVariableByExtraColumns[extraCols]]
        ExtraColsVsMethodsAndListOfFeatures[str(extraCols)]=MethodVsListOfFeatures
    print("Number of Extracols is ",extraCols," and ExtraColsVsMethodsAndListOfFeatures:\n",ExtraColsVsMethodsAndListOfFeatures)
    
    file_name = "/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/CriteoRandIntLR_trial2LR__OtherMethodsScores_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE"+str(foldNum)+".pkl"
    open_file = open(file_name, "wb")

    pickle.dump(ExtraColsVsMethodsAndListOfFeatures, open_file)

    open_file.close()
    ############################################################################################################################################################
    print("Starting learning and testing for the folds")
    pool = mp.Pool(processes=30)
    
    ArgumentsToPass=[]
    for extraCols in UnnecessaryCols:
        ArgumentsToPass.append([df_train,df_test,extraCols])
    QINI_SCORES_BY_METHOD_UNNECESSARYCOLS = pool.map(LearnInParallelWithDifferentUnnecessaryCols, ArgumentsToPass)
    pool.close()
    FINALRESULTS.append(QINI_SCORES_BY_METHOD_UNNECESSARYCOLS)
#     file_name = "~/../../data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/CriteoRandIntLR_trial2LR__FINALRESULT_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE"+str(foldNum)+".pkl"
    file_name = "/data/userstorage/mXXXXXXXX/Datasets/featureSeleExpProtocol/criteo/criteo_IndTAndC_Folds/CriteoRandIntLR_trial2LR__FINALRESULT_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE"+str(foldNum)+".pkl"
    open_file = open(file_name, "wb")

    pickle.dump(FINALRESULTS, open_file)

    open_file.close()

    
file_name = "CriteoRandIntLR_trial2LR__FINALRESULT_EXPERIMENTS_ED_192Unnecessary_FEATURESELECTION_MixedNOISE.pkl"
print("finished, saving now")
open_file = open(file_name, "wb")

pickle.dump(FINALRESULTS, open_file)

open_file.close()

stop_counter(1)

for i in range(_NumberOfCounters):
    print("Exp Counter ",i)
    print("Exp delta time counter ",_deltatime_counter[i])
    print("Exp _nb_counter ",_nb_counter[i])


    