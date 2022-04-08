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
import sys
import random



def createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl):
    TreatmentGroup["Y"]=choices([0,1], [1-OutputProportionTrt,OutputProportionTrt],k=TreatmentGroup.shape[0])
    ControlGroup["Y"]=choices([0,1], [1-OutputProportionCtrl,OutputProportionCtrl],k=ControlGroup.shape[0])

    TreatmentGroup["CATE"]=OutputProportionTrt-OutputProportionCtrl
    ControlGroup["CATE"]=OutputProportionTrt-OutputProportionCtrl
    
    TreatmentGroup['ProbY_trt']=OutputProportionTrt
    TreatmentGroup['ProbY_ctrl']=OutputProportionCtrl
    ControlGroup['ProbY_trt']=OutputProportionTrt
    ControlGroup['ProbY_ctrl']=OutputProportionCtrl
    
    return TreatmentGroup,ControlGroup

def createTreatmentAndControlGroupsWithContinuousCATE(TreatmentGroup,ControlGroup,OutputProportionCtrl):
    
    
    TreatmentGroup["Y"] = TreatmentGroup['X'].apply(lambda x: 1 if random.uniform(0, 1) < x/10 else 0)
    
    ControlGroup["Y"]=choices([0,1], [1-OutputProportionCtrl,OutputProportionCtrl],k=ControlGroup.shape[0])

    TreatmentGroup["CATE"]=TreatmentGroup["X"]/10-OutputProportionCtrl
    ControlGroup["CATE"]=ControlGroup["X"]/10-OutputProportionCtrl
    
    TreatmentGroup['ProbY_trt']=TreatmentGroup["X"]/10
    TreatmentGroup['ProbY_ctrl']=OutputProportionCtrl
    ControlGroup['ProbY_trt']=ControlGroup["X"]/10
    ControlGroup['ProbY_ctrl']=OutputProportionCtrl
    
    return TreatmentGroup,ControlGroup

# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================

# ======================================================================================================================================
def generatingTreatmentSineControlCosinePattern(DataSize_per_Interval,TreatmentGroup,ControlGroup,intervalIndex,TotalIntervalNUM=10):
    OutputProportionTrt=0.5+(0.5*np.sin(intervalIndex*2*math.pi/TotalIntervalNUM))
    OutputProportionCtrl=0.5+(0.5*np.cos(intervalIndex*2*math.pi/TotalIntervalNUM))

    
    
    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generatingContinuousAscending(DataSize_per_Interval,TreatmentGroup,ControlGroup,OutputProbabilityInControl=0.5):
    OutputProportionCtrl=OutputProbabilityInControl

    
    
    return createTreatmentAndControlGroupsWithContinuousCATE(TreatmentGroup,ControlGroup,OutputProportionCtrl)

# ======================================================================================================================================
def generatingPureCATE_Positive_Zero(DataSize_per_Interval,TreatmentGroup,ControlGroup,PositiveInterval=False,Purity=1):
    if PositiveInterval:
        OutputProportionTrt=Purity
        OutputProportionCtrl=1-Purity
    else:
        OutputProportionTrt=0.5
        OutputProportionCtrl=0.5
    
    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generatingCATE_Positive_Negative(DataSize_per_Interval,TreatmentGroup,ControlGroup,PositiveInterval=False,Purity=0.8):
    if PositiveInterval:
        OutputProportionTrt=Purity
        OutputProportionCtrl=1-Purity
    else:
        OutputProportionTrt=1-Purity
        OutputProportionCtrl=Purity
    
    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generatingAscendingCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,intervalIndex,OutputProbabilityInControl=0.5,TotalIntervalNUM=10):
    OutputProportionTrt=intervalIndex/TotalIntervalNUM
    OutputProportionCtrl=OutputProbabilityInControl

    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generatingDescendingCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,intervalIndex,OutputProbabilityInControl=0.5,TotalIntervalNUM=10):
    OutputProportionTrt=1-intervalIndex/TotalIntervalNUM
    OutputProportionCtrl=OutputProbabilityInControl

    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generateXCATE(DataSize_per_Interval,TreatmentGroup,ControlGroup,intervalIndex,TotalIntervalNUM=10):
    OutputProportionTrt=1-intervalIndex/TotalIntervalNUM
    OutputProportionCtrl=intervalIndex/TotalIntervalNUM

    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
def generateInverseXInversed(DataSize_per_Interval,TreatmentGroup,ControlGroup,intervalIndex,TotalIntervalNUM=10):
    if intervalIndex<TotalIntervalNUM/2:
        OutputProportionTrt=1-intervalIndex/TotalIntervalNUM
        OutputProportionCtrl=intervalIndex/TotalIntervalNUM
    else:
        OutputProportionTrt=intervalIndex/TotalIntervalNUM
        OutputProportionCtrl=1-intervalIndex/TotalIntervalNUM
    
    return createTreatmentAndControlGroupsWithSpecificCATE(TreatmentGroup,ControlGroup,OutputProportionTrt,OutputProportionCtrl)
# ======================================================================================================================================
