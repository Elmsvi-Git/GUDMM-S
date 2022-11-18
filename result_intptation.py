# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:45:02 2020

@author: elahe
"""
import numpy as np
import pandas as pd
from Huangarian_accuracy import haungarian_accuracy

from sklearn.metrics.cluster import adjusted_rand_score , rand_score

from internal_validation import internalIndex
from sklearn import metrics   
from sklearn.metrics.cluster import normalized_mutual_info_score , calinski_harabasz_score

from sklearn.preprocessing import MinMaxScaler
def result_intprtn2(label, y_true ,X , no_fgid='all'):
    
    Data = X.copy()    
    sc = MinMaxScaler()
    Data = sc.fit_transform(Data)
       
    criteria =['ACC', 'RI' , 'NMI' ,'I', 'calinski','xieBenie','davies','SDbw']
    results = pd.DataFrame(data = np.zeros([1 , 8]) ,  columns =  criteria)

    no_clusters = int(max(label))
    
    if(no_fgid =='one'):
        true_no_clusters = int(np.max(y_true))
    else:
        true_no_clusters = int(y_true.shape[1])
        
    Cl_intpt = np.zeros([no_clusters,true_no_clusters])

    from sklearn import metrics
    if(no_fgid=='one'): 
        if(true_no_clusters==no_clusters):
            results['ACC'] = round(haungarian_accuracy(y_true-1  , label.ravel()-1) , 4)
        # results['ARI'] = round(adjusted_rand_score(label, y_true) , 3)
        results['RI'] = round(rand_score(label, y_true) , 4)
        results['NMI'] = round(normalized_mutual_info_score(label, y_true) , 3)
    
    for i in range(no_clusters):
        if (np.sum(label==i+1)==0):
            print('Colul not find ', no_clusters, ' clusters')
            return results , Cl_intpt
    IntInd   = internalIndex(no_clusters)  
    results['I'] = round(IntInd.I(Data, label),4)
    results['calinski'] =round(calinski_harabasz_score(Data, label) , 4)

    results['xieBenie'] = round(IntInd.xie_benie(Data, label),4)
    results['davies'] = round(metrics.davies_bouldin_score(Data, label) , 4)
    results['SDbw'] = round(IntInd.SDbw(Data, label) , 4)    
        
        
    if(no_fgid =='one'):
        for i in range(1 , no_clusters+1):
            cluster_i = label==i
            for j in range(1 , true_no_clusters+1):
                Cl_intpt[i-1 ,j-1] = int(np.sum(y_true[cluster_i] ==j))
                # print('no. of ind. with fgid' , j , ' in cluster ', i ,' is: ' , Cl_intpt[i-1 , j-1])

    else:#(no_fgid =='all') :
        for i in range(1 , no_clusters+1):
            cluster_i = label==i
            for j in range(true_no_clusters):
                dis_val_j = y_true[:,j]#y_true.iloc[:,j]
                Cl_intpt[i-1 ,j] = int(np.sum(dis_val_j[cluster_i]))
 
    return results , Cl_intpt


