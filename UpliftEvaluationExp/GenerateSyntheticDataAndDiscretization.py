import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import numpy as np
from random import seed
from random import randint
from random import gauss
from random import uniform
from random import shuffle
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib qt5
from sklearn.model_selection import train_test_split
import hashlib
import random
random.seed(6)#my main seed
from IPython.display import Image
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.metrics import plot_gain
from causalml.metrics import get_qini
from causalml.metrics import plot_qini
from causalml.metrics import qini_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from itertools import permutations
import os
import sys
random.seed(6)
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)
from sklift.viz import plot_qini_curve
# import approaches
from sklift.models import SoloModel, ClassTransformation, TwoModels
# import any estimator adheres to scikit-learn conventions.
from catboost import CatBoostClassifier
import numpy as np; np.random.seed(0)

import seaborn as sns; sns.set_theme()

from IPython.display import Image
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
import math
from random import choices

import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import scipy.stats

import numpy as np
import pandas as pd
from causalml.dataset import make_uplift_classification
from causalml.feature_selection.filters import FilterSelect
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.metrics import plot_gain, auuc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

import pandas as pd
from causalml.feature_selection.filters import FilterSelect
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.metrics import plot_gain, auuc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rcParams

from operator import itemgetter

from math import log, pi, pow, exp, lgamma, sqrt
import numpy as np
from typing import Callable
from math import ceil, floor
from operator import itemgetter
from sortedcontainers import SortedKeyList
from operator import add
import numpy as np
import pandas as pd
import logging
import time
import os
import pickle
import sys

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from scipy.special import comb
import scipy.special as sc

from operator import itemgetter
import operator
import bisect

import multiprocessing as mp
import greedySearchLinkedListWithSplitPostOpt
import sys


from Patterns import generatingTreatmentSineControlCosinePattern
from Patterns import generatingPureCATE_Positive_Zero
from Patterns import generatingCATE_Positive_Negative
from Patterns import generatingAscendingCATE
from Patterns import generatingDescendingCATE
from Patterns import generateXCATE
from Patterns import generateInverseXInversed
from Patterns import generatingContinuousAscending



'''
[CATE_Positive_Negative , PureCATE_Positive_Zero , TreatmentSineControlCosinePattern , AscendingCATE , DescendingCATE , X , generateInverseXInversed]
'''
patterns=[ 'CATE_Positive_Negative','PureCATE_Positive_Zero' , 'TreatmentSineControlCosinePattern' , 'AscendingCATE' , 'DescendingCATE' , 'X' , 'generateInverseXInversed','ContinousAscending']
# patterns=['TreatmentSineControlCosinePattern']

# patterns=['CATE_Positive_Negative' , 'PureCATE_Positive_Zero' , 'TreatmentSineControlCosinePattern' , 'DescendingCATE' , 'X' , 'generateInverseXInversed']
# patterns=[ 'AscendingCATE' ,'TreatmentSineControlCosinePattern' , 'DescendingCATE' , 'X' , 'generateInverseXInversed']
# patterns=[ 'AscendingCATE']
# patterns=['CATE_Positive_Negative' , 'PureCATE_Positive_Zero']
# patterns=['ContinousAscending']
GeneratingTestData=False

Purity=0.6
BalancedTreatmentAndControlGroupsSize=True

if BalancedTreatmentAndControlGroupsSize==False:
    balanced="NotBALANCED"
else:
    balanced=''

for pattern in patterns:
    
    old_stdout = sys.stdout
    log_file = open("./"+pattern+"_logFile.log","w")
    sys.stdout = log_file


    IntervalsNUM=10

    DATA_PATH="~/../../data/userstorage/mrafla/Datasets/SyntheticDatasetsForMODL/"+pattern
    test_data_path="./TestDataForDifferentPatterns/"+pattern

    if GeneratingTestData==False: 
        interval_size_list=list(np.logspace(0, 4, num=40,base=10))
        interval_size_list=[floor(el) for el in interval_size_list]
        print(interval_size_list)
    else:
        interval_size_list=[1000]

    for DataSize_per_Interval in interval_size_list:
        for version in range(1,IntervalsNUM+1):
            random.seed(version*556)
            cols=['X','Y','T','CATE','ProbY_trt','ProbY_ctrl']
            MainTrtGroup=pd.DataFrame(columns=cols)
            MainCtrlGroup=pd.DataFrame(columns=cols)

            PositiveInterval=False
            
            AllData=pd.DataFrame({'X':np.random.uniform(0, IntervalsNUM, size=(1, DataSize_per_Interval*IntervalsNUM))[0]})
            for i in range(0,IntervalsNUM):
                data=AllData[(AllData['X']>=i)&(AllData['X']<=i+1)]
#                 data=pd.DataFrame({'X':np.random.uniform(i, i+1, size=(1, DataSize_per_Interval))[0]})
                
                if BalancedTreatmentAndControlGroupsSize:
                    TreatmentControlRandomValues=[random.randint(0, 1) for _ in range(data.shape[0])]
                else:
                    TreatmentControlRandomValues=np.random.choice([0,1], p=[0.75,0.25],size=data.shape[0])
                data['T']=TreatmentControlRandomValues

                TreatmentGroup=pd.DataFrame(data[data['T']==1].values.tolist(),columns=["X",'T'])
                ControlGroup=pd.DataFrame(data[data['T']==0].values.tolist(),columns=["X",'T'])

                #Checking the pattern to be generated
                if pattern=="TreatmentSineControlCosinePattern":
                    TreatmentAndControlGroups=generatingTreatmentSineControlCosinePattern(DataSize_per_Interval,TreatmentGroup,ControlGroup,i)
                elif pattern=="PureCATE_Positive_Zero":
                    if PositiveInterval==False:
                        PositiveInterval=True
                    else:
                        PositiveInterval=False
                    TreatmentAndControlGroups=generatingPureCATE_Positive_Zero(DataSize_per_Interval,TreatmentGroup,ControlGroup,PositiveInterval,Purity)
                elif pattern=="CATE_Positive_Negative":
                    if PositiveInterval==False:
                        PositiveInterval=True
                    else:
                        PositiveInterval=False
                    TreatmentAndControlGroups=generatingCATE_Positive_Negative(DataSize_per_Interval,TreatmentGroup,ControlGroup,PositiveInterval,Purity)
                elif pattern=='AscendingCATE':
                    TreatmentAndControlGroups=generatingAscendingCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,i)
                elif pattern=='DescendingCATE':
                    TreatmentAndControlGroups=generatingDescendingCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,i)
                elif pattern=='X':
                    TreatmentAndControlGroups=generateXCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,i,IntervalsNUM)
                elif pattern=='generateInverseXInversed':
                    TreatmentAndControlGroups=generateInverseXInversed(DataSize_per_Interval,TreatmentGroup,ControlGroup,i,IntervalsNUM)
                elif pattern=="ContinousAscending":
                    TreatmentAndControlGroups=generatingContinuousAscending(DataSize_per_Interval,TreatmentGroup,ControlGroup)

                MainTrtGroup=MainTrtGroup.append(TreatmentAndControlGroups[0], ignore_index=True)
                MainCtrlGroup=MainCtrlGroup.append(TreatmentAndControlGroups[1], ignore_index=True)

            objs=[MainTrtGroup,MainCtrlGroup]
            df = pd.concat(objs)

            columns_titles = ["X","T","Y","CATE",'ProbY_trt','ProbY_ctrl']
            df=df.reindex(columns=columns_titles)

            if GeneratingTestData==False: 
                if pattern=="PureCATE_Positive_Zero" or pattern=="CATE_Positive_Negative":
                    p='_'+str(Purity)
                else:
                    p=''
                df.to_csv(DATA_PATH+str(DataSize_per_Interval*10)+"_"+str(version)+p+balanced+".csv")
            else:
                if pattern=="PureCATE_Positive_Zero" or pattern=="CATE_Positive_Negative":
                    p='_'+str(Purity)
                else:
                    p=''
                df.to_csv(test_data_path+p+balanced+".csv")

    if GeneratingTestData==False: 
        dataPath=DATA_PATH
        AllDataPaths=[]
        for i in interval_size_list:
            for version in range(1,IntervalsNUM+1):
                if pattern=="PureCATE_Positive_Zero" or pattern=="CATE_Positive_Negative":
                    p='_'+str(Purity)
                else:
                    p=''
                AllDataPaths.append(dataPath+str(i*IntervalsNUM)+"_"+str(version)+p+balanced+".csv")

#         pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(processes=10)
#         pool = mp.Semaphore(multiprocessing.cpu_count())
        if pattern=="PureCATE_Positive_Zero" or pattern=="CATE_Positive_Negative":
            p='_'+str(Purity)
        else:
            p=''
        
        args=[]
        for element in AllDataPaths:
            args.append([element,test_data_path+p+balanced+".csv"])
        result = pool.map(greedySearchLinkedListWithSplitPostOpt.ExecuteGreedySearchAndPostOpt, args)

        print("finished now saving file")

        file_name = "./PKL_files/RESULT"+pattern+p+balanced+"SplitAtTheEND.pkl"


        open_file = open(file_name, "wb")

        pickle.dump(result, open_file)

        open_file.close()

