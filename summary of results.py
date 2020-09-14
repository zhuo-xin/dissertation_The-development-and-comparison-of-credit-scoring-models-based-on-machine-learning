# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:49:01 2020

@author: 35453
"""

## SUMMARY OF RESULTS

## obtain confusion matrixes of all algorithms
algorithm_cm_list=[cm_DT,cm_SVM,cm_KNN,cm_NB,cm_LR,cm_MLP,
                   cm_RF,cm_ADT,cm_ALR,cm_GBDT,
                   cm_ST1,cm_ST2,cm_ST3,cm_ST4]
TP_list=list(range(14))
FP_list=list(range(14))
FN_list=list(range(14))
TN_list=list(range(14))
m=0
for i in algorithm_cm_list:
    TP_list[m]=i[0][0]
    FP_list[m]=i[0][1]
    FN_list[m]=i[1][0]
    TN_list[m]=i[1][1]
    m=m+1
    
c={"name" : algorithm_name_list,
   "TP" : TP_list,
   "FP" : FP_list,
   "FN" : FN_list,
   "TN" : TN_list
   }
import pandas as pd
df=pd.DataFrame(c)
df                       ## confusion matrix of all algorithms
df.to_csv(r'confusion matrix of all algorithms.csv', index=False)   

## TP bar chart
df=df.sort_values(by="TP" , ascending=False)
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([600, 1150])
plt.xlabel("algorithm")
plt.ylabel("TP value")
plt.title("TP bar chart")
plt.bar(df['name'], df['TP'],color='thistle',tick_label=df['name'])
plt.show()

## TN bar chart
df=df.sort_values(by="TN" , ascending=False)
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([5700, 6500])
plt.xlabel("algorithm")
plt.ylabel("TN value")
plt.title("TN bar chart")
plt.bar(df['name'], df['TN'],color='tan',tick_label=df['name'])
plt.show()

## find the algorithm with the highest accuracy    
algorithm_name_list=['DT','SVM','KNN','NB','LR','MLP',
                     'RF','ADT','ALR','GBDT',
                     'ST1','ST2','ST3','ST4']
algorithm_accuracy_list=[accuracy_score_DT,accuracy_score_SVM,accuracy_score_KNN,accuracy_score_NB,
                         accuracy_score_LR,accuracy_score_MLP,accuracy_score_RF,accuracy_score_ADT,
                         accuracy_score_ALR,accuracy_score_GBDT,accuracy_score_ST1,accuracy_score_ST2,
                         accuracy_score_ST3,accuracy_score_ST4]


c={"name" : algorithm_name_list,
   "value" : algorithm_accuracy_list}
import pandas as pd
df=pd.DataFrame(c)
df=df.sort_values(by="value" , ascending=False)
df
df.to_csv(r'accuracy of all algorithms.csv', index=False)

# draw bar chart of accuracy of all algorithms
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.75, 0.95])
plt.xlabel("algorithm")
plt.ylabel("accuracy")
plt.title("accuracy bar chart")
plt.bar(df['name'], df['value'],color='thistle',tick_label=df['name'])
plt.show()

# draw accuracy bar chart of stacking
df_ST_accuracy=df.iloc[0:4,:]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.9387, 0.940])
plt.xlabel("algorithm")
plt.ylabel("accuracy")
plt.title("accuracy bar chart of stackings")
plt.bar(df_ST_accuracy['name'], df_ST_accuracy['value'],color='thistle',tick_label=df['name'])
plt.show()


## find the algorithm with the highest precision
algorithm_precision_list=[precision_score_DT,precision_score_SVM,precision_score_KNN,precision_score_NB,
                         precision_score_LR,precision_score_MLP,precision_score_RF,precision_score_ADT,
                         precision_score_ALR,precision_score_GBDT,precision_score_ST1,
                         precision_score_ST2,precision_score_ST3,precision_score_ST4]

c={"name" : algorithm_name_list,
   "value" : algorithm_precision_list}
import pandas as pd
df=pd.DataFrame(c)
df=df.sort_values(by="value" , ascending=False)
df
df.to_csv(r'precision of all algorithms.csv', index=False)


# draw bar chart of precision of all algorithms
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.75, 0.95])
plt.xlabel("algorithm")
plt.ylabel("precision")
plt.title("precision bar chart")
plt.bar(df['name'], df['value'],color='grey',tick_label=df['name'])
plt.show()

# draw precision bar chart of stacking
df_ST_precision=df.iloc[0:4,:]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.939, 0.941])
plt.xlabel("algorithm")
plt.ylabel("precision")
plt.title("precision bar chart of stackings")
plt.bar(df_ST_precision['name'], df_ST_precision['value'],color='grey',tick_label=df['name'])
plt.show()


## find the algorithm with the highest f1
algorithm_f1_list=[f1_score_DT,f1_score_SVM,f1_score_KNN,f1_score_NB,
                         f1_score_LR,f1_score_MLP,f1_score_RF,f1_score_ADT,
                         f1_score_ALR,f1_score_GBDT,f1_score_ST1,
                         f1_score_ST2,f1_score_ST3,f1_score_ST4]

c={"name" : algorithm_name_list,
   "value" : algorithm_f1_list}
import pandas as pd
df=pd.DataFrame(c)
df=df.sort_values(by="value" , ascending=False)
df
df.to_csv(r'f1 score of all algorithms.csv', index=False)

# draw bar chart of f1 with different algorithms
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.75, 0.95])
plt.xlabel("algorithm")
plt.ylabel("f1")
plt.title("f1 score bar chart")
plt.bar(df['name'], df['value'],color='tan',tick_label=df['name'])
plt.show()

# draw f1 bar chart of stacking
df_ST_f1=df.iloc[0:4,:]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.935, 0.937])
plt.xlabel("algorithm")
plt.ylabel("f1")
plt.title("f1 bar chart of stackings")
plt.bar(df_ST_f1['name'], df_ST_f1['value'],color='tan',tick_label=df['name'])
plt.show()


## find the algorithm with the highest recall
algorithm_recall_list=[recall_score_DT,recall_score_SVM,recall_score_KNN,recall_score_NB,
                         recall_score_LR,recall_score_MLP,recall_score_RF,recall_score_ADT,
                         recall_score_ALR,recall_score_GBDT,recall_score_ST1,
                         recall_score_ST2,recall_score_ST3,recall_score_ST4]

c={"name" : algorithm_name_list,
   "value" : algorithm_recall_list}
import pandas as pd
df=pd.DataFrame(c)
df=df.sort_values(by="value" , ascending=False)
df
df.to_csv(r'recall of all algorithms.csv', index=False)

# draw bar chart of f1 with different algorithms
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.75, 0.95])
plt.xlabel("algorithm")
plt.ylabel("recall value")
plt.title("recall bar chart")
plt.bar(df['name'], df['value'],color='peru',tick_label=df['name'])
plt.show()

# draw recall bar chart of stacking
df_ST_recall=df.iloc[0:4,:]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.93875, 0.940])
plt.xlabel("algorithm")
plt.ylabel("recall")
plt.title("recall bar chart of stackings")
plt.bar(df_ST_recall['name'], df_ST_recall['value'],color='peru',tick_label=df['name'])
plt.show()

## find the algorithm with the highest roc_auc
algorithm_roc_auc_list=[roc_auc_DT,roc_auc_SVM,roc_auc_KNN,roc_auc_NB,
                         roc_auc_LR,roc_auc_MLP,roc_auc_RF,roc_auc_ADT,
                         roc_auc_ALR,roc_auc_GBDT,roc_auc_ST1,
                         roc_auc_ST2,roc_auc_ST3,roc_auc_ST4]

c={"name" : algorithm_name_list,
   "value" : algorithm_roc_auc_list}
import pandas as pd
df=pd.DataFrame(c)
df=df.sort_values(by="value" , ascending=False)
df
df.to_csv(r'AUC of all algorithms.csv', index=False)

# draw bar chart of auc with different algorithms
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.75, 0.95])
plt.xlabel("algorithm")
plt.ylabel("AUC value")
plt.title("AUC bar chart")
plt.bar(df['name'], df['value'],color='salmon',tick_label=df['name'])
plt.show()

# draw ensemble learning bar chart
df_EL_auc=df.iloc[0:7,:]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.925, 0.94])
plt.xlabel("algorithm")
plt.ylabel("AUC value")
plt.title("AUC bar chart of ensenmble learning")
plt.bar(df_EL_auc['name'], df_EL_auc['value'],color='salmon',tick_label=df['name'])
plt.show()

# draw stackings bar chart
df_ST_auc=df_EL_auc.iloc[[0,2,3,4]]
plt.figure()
plt.figure(figsize=(10,5))
plt.ylim([0.937, 0.94])
plt.xlabel("algorithm")
plt.ylabel("AUC value")
plt.title("AUC bar chart of stackings")
plt.bar(df_ST_auc['name'], df_ST_auc['value'],color='salmon',tick_label=df['name'])
plt.show()