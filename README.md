# A Non-Parameteric-Bayesian-Approach-for-Uplift-Discretization-and-Feature-Selection

This repository contains the code of the paper entitled : Non-Parameteric Bayesian Approach for Uplift Discretization and Feature Selection submitted to  the conference ECML/PKDD 2022.

## UMODL Search algorithm CODE
Greedy Search algorithm for UMODL criterion

### How to use it ?

Explanatory variable should be numerical, binary treatment variable {0,1}, and binary outcome variable {0,1}.

<pre><code>
from UMODL_SearchAlgorithm import ExecuteGreedySearchAndPostOpt
import pandas as pd

df = pd.read_csv(data.csv)
feature=df[['Variable','treatment','outcome']]
feature_level,DiscretizationBounds=ExecuteGreedySearchAndPostOpt(feature)

</code></pre>


## Requirements
Python 3.7

## Feature selection experiments
After runnning the file **FeatureSelectionExperimentalProtocol.py** the following files will be generated:

**FINALRESULT10.pkl** will contain all the qini results from the experimental protocol for feature selection for all folds for feature selection methods.

**MODL_Scores.pkl** will contain the UMODL scores for each variable in each fold. Noisy variables will always have zero scores and ignored in the uplift modeling process.

**OtherScores.pkl** will contain the Top M variables selected by each of the state-of-art feature selection methods, where M are the number of non-zero scored variables by UMODL.



## Supplementary material

![image](https://user-images.githubusercontent.com/75427835/162019101-ebcebd91-907a-43a7-ad2a-12267836cc24.png)
![image](https://user-images.githubusercontent.com/75427835/162020092-4f0471c4-9aee-4865-ad92-265a59a6896c.png)

==========================================================================================

![image](https://user-images.githubusercontent.com/75427835/162019028-562a0624-7478-46f8-a7d1-3ca704c9b3a3.png)![image](https://user-images.githubusercontent.com/75427835/162020198-d80a21fe-9f2b-42d6-a0af-9434fcf1deb0.png)


==========================================================================================

![image](https://user-images.githubusercontent.com/75427835/162023603-1abbe527-bfaf-4d7e-942b-bd15d52e6ac6.png)![image](https://user-images.githubusercontent.com/75427835/162020414-1130e885-829f-4316-93c1-48c0613babab.png)

==========================================================================================
### Unbalanced treatment and control groups

![image](https://user-images.githubusercontent.com/75427835/162019221-8bbb4d59-dabc-42c0-ad9c-2699188c8475.png)
![image](https://user-images.githubusercontent.com/75427835/162020546-ed029145-a4a0-4f64-83ab-f753f5614cd3.png)

==========================================================================================

### Unbalanced treatment and control groups


![image](https://user-images.githubusercontent.com/75427835/162019167-3e19591f-d93f-4051-9a5d-b83a1679280e.png)
![image](https://user-images.githubusercontent.com/75427835/162020509-070b04be-f473-4833-81b2-51585237a311.png)






