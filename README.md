# A Non-Parameteric-Bayesian-Approach-for-Uplift-Discretization-and-Feature-Selection

This repository contains the code of the paper entitled : Non-Parameteric Bayesian Approach for Uplift Discretization and Feature Selection submitted to  the conference ECML/PKDD 2022.

### Abstract:
Uplift modeling aims to estimate the incremental impact of a treatment, such as a marketing campaign or a drug, on an individual’s outcome. Uplift data of Bank or Telecom have often hundreds to thousands of features. In such situations, detection of irrelevant features is an essential step to reduce computational time and increase model performance. We present a parameter-free feature selection method for uplift modeling founded on a Bayesian approach. We start by defining UMODL an automatic feature discretization method for uplift. UMODL is based on a space of discretization models and a prior distribution. From this model space, we define a Bayes optimal evaluation criterion of a discretization model for uplift.  We then propose a O(n log n) optimization algorithm that finds near-optimal discretization for estimating uplift. Experiments demonstrate the high performances obtained by this new discretization method. Then we describe UMODL feature selection a parameter-free feature selection method for uplift. Experiments show that the new method both removes irrelevant features and achieves better performances than state of the art methods.

### CODE
The code consists of:

1. Greedy Search algorithm with post optimization
2. Experiments for UMODL evaluation
3. Experiments for UMODL feature selection


### Requirements
Python 3.7

### Supplementary material
![image](https://user-images.githubusercontent.com/75427835/162019101-ebcebd91-907a-43a7-ad2a-12267836cc24.png)
![image](https://user-images.githubusercontent.com/75427835/162020092-4f0471c4-9aee-4865-ad92-265a59a6896c.png)

==========================================================================================
![image](https://user-images.githubusercontent.com/75427835/162019028-562a0624-7478-46f8-a7d1-3ca704c9b3a3.png)![image](https://user-images.githubusercontent.com/75427835/162020198-d80a21fe-9f2b-42d6-a0af-9434fcf1deb0.png)


==========================================================================================
![image](https://user-images.githubusercontent.com/75427835/162023603-1abbe527-bfaf-4d7e-942b-bd15d52e6ac6.png)![image](https://user-images.githubusercontent.com/75427835/162020414-1130e885-829f-4316-93c1-48c0613babab.png)

==========================================================================================
Positive Negative UNBALANCED

![image](https://user-images.githubusercontent.com/75427835/162019221-8bbb4d59-dabc-42c0-ad9c-2699188c8475.png)
![image](https://user-images.githubusercontent.com/75427835/162020546-ed029145-a4a0-4f64-83ab-f753f5614cd3.png)

==========================================================================================

Positive Zero UNBALANCED

![image](https://user-images.githubusercontent.com/75427835/162019167-3e19591f-d93f-4051-9a5d-b83a1679280e.png)
![image](https://user-images.githubusercontent.com/75427835/162020509-070b04be-f473-4833-81b2-51585237a311.png)







