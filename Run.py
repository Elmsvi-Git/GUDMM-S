# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 22:36:28 2022

@author: ElaheMsvi

"""
import numpy as np
from public_data import public_data_calling
from result_intptation import  result_intprtn2
from sklearn.cluster import SpectralClustering
from GUDMM import ditance_dependency_mixed_matrix
#------------------------------------------------------------------------
#data_name = ['celev_heart','credit','dermatology','german','adault2_dep','australian', 
#             'soybean_small','vote', 'breast','car','LG','MM',
#             'iris','wine','ion','sonar','yeast','ecoli']


data_name = 'celev_heart'
X , y, K , N ,no_f, no_f_cont , no_f_ord = public_data_calling(name = 
                                                               data_name)#, type_data='pure_continues')
X[:, no_f_cont:] = X[:, no_f_cont:].astype('int')


no_f_disc = no_f - no_f_cont


#-------------------------------------------------------------------
#           10 Iteration of GUDMM
#-------------------------------------------------------------------

DM = 'DM5'
j = 0 
no_iteration =50
results = np.zeros([no_iteration+2 , 8])
import time
start_time = time.time()
for itr in range(no_iteration):
    dist = ditance_dependency_mixed_matrix(X ,no_f_cont , no_f_ord , DM)
    #------------------------------
    B = 10#Heart , Credit: 0.10
    affinity_X = np.exp(- B*dist** 2 )
    #------------------------------
    # sigma = np.mean(dist , axis = 1)
    #------------------------------
    # sigma = np.sort(dist , axis = 1)[:,7]#iris
    #------------------------------
    # g = int(N/(K**2))
    # sigma = np.mean(np.sort(dist , axis = 1)[:,:g] , axis = 1)
    # #------------------------------
    # sigma_all = np.sort(dist ,axis = 1)
    # sigma = np.zeros([N,1])
    # for ii in range(N):
    #     id_nonzero = next((i for i, x in enumerate(sigma_all[ii,:]) if x), None)
    #     sigma[ii] = sigma_all[ii, id_nonzero+6]
    # #------------------------------
    # Sigma_mat = np.matmul(sigma.reshape(-1,1),sigma.reshape(1,-1))
    # affinity_X = np.exp(- np.divide((dist** 2),Sigma_mat ))
    #------------------------------
   
    clustering = SpectralClustering(affinity='precomputed',n_clusters=K, assign_labels="kmeans")#,random_state=0),  
    labels = clustering.fit_predict(affinity_X)+1

    results_df, _ = result_intprtn2(labels, y ,X ,'one')       
    results[j , :] = results_df.values
    j = j+1

results[-2 , :] = np.round(np.mean(results[:-2 , :] , axis =0) , 4)
results[-1 , :] = np.round(np.std( results[:-2 , :] , axis =0) , 4)
print('Ex. Time:' ,np.round((time.time()-start_time)/(no_iteration) , 4))
#-------------------------------------------------------------------
#                               Time
#--------------------------------------------------
'''
DM = 'DM5'
j = 0 
results = np.zeros([18 , 1])
import time
#data_list = ['celev_heart',
#             'credit',
#             'dermatology','german',
#             'adault2_dep','australian', 
#             'soybean_small','vote', 'breast','car','LG','MM',
#             'ion','sonar','yeast','ecoli'
#             'iris','wine'
]

i  = 0
for data_name in data_list:
    X , y, K , N ,no_f, no_f_cont , no_f_ord = public_data_calling(name = 
                                                                   data_name)#, type_data='pure_continues')
    X[:, no_f_cont:] = X[:, no_f_cont:].astype('int')   
    no_f_disc = no_f - no_f_cont
    min_sample = 1
    n_sampling = 1001      
    sc =  MinMaxScaler()#Normalizer()#StandardScaler()
    if(no_f_cont):
        X[:,:no_f_cont ] = sc.fit_transform(X[:,:no_f_cont ])
    X_sc = sc.fit_transform(X)
    start_time = time.time()
    dist = ditance_dependency_mixed_matrix(X ,no_f_cont , no_f_ord , DM)
    #------------------------------
    # B = .1
    # affinity_X = np.exp(- B*dist** 2 )
    #------------------------------
    # sigma = np.mean(dist , axis = 1)
    #------------------------------
    # sigma = np.sort(dist , axis = 1)[:,7]#derma,iris
    #------------------------------
    # g = int(N/(K**2))
    # sigma = np.mean(np.sort(dist , axis = 1)[:,:g] , axis = 1)#derma,iris
    #------------------------------
    sigma_all = np.sort(dist ,axis = 1)
    sigma = np.zeros([N,1])
    for ii in range(N):
        id_nonzero = next((i for i, x in enumerate(sigma_all[ii,:]) if x), None)
        sigma[ii] = sigma_all[ii, id_nonzero+6]
    #------------------------------
    Sigma_mat = np.matmul(sigma.reshape(-1,1),sigma.reshape(1,-1))
    affinity_X = np.exp(- np.divide((dist** 2),Sigma_mat ))
    #------------------------------
    clustering = SpectralClustering(affinity='precomputed',n_clusters=K, assign_labels="kmeans")#,random_state=0),  
    labels = clustering.fit_predict(affinity_X)+1
    results[i] = np.round((time.time()-start_time),2)
    i+=1
    

'''

#---------------------------------------------
#           Selection of B2
#--------------------------------------------
'''
# just used for german and ecoli
DM = 'DM5'
j = 0 
results = np.zeros([9 , 8])
dist = ditance_dependency_mixed_matrix(X ,no_f_cont , no_f_ord , DM)
for B in [.01 , .1 , .5 , 1 , 2 , 5 , 10 , 12 , 15]:# , 12 , 15 , 20range(K,K+1):
    affinity_X = np.exp(- B*dist** 2 )
    clustering = SpectralClustering(affinity='precomputed',n_clusters=K, assign_labels="kmeans")
    labels = clustering.fit_predict(affinity_X)+1
    results_df,_ = result_intprtn2(labels, y ,X_sc ,'one')       
    results[j , :] = results_df.values
    j = j+1
'''

#---------------------------------------------------
#       Test of different DM
#---------------------------------------------------

'''
# B = .1
j = 0 
results = np.zeros([6 , 8])

for DM in [ 'DM5']:#'DM0', 'DM1' ,'DM2' ,'DM3','DM4' , 
    # B = .5
    dist = ditance_dependency_mixed_matrix(X ,no_f_cont , no_f_ord , DM)
    # affinity_X = np.exp(- B*dist** 2 )
    #------------------------------------
    sigma = np.mean(dist , axis = 1)
    #------------------------------------
    # sigma = np.sort(dist , axis = 1)[:,7]#derma,iris,#heart can be too
    #------------------------------------
    Sigma_mat = np.matmul(sigma.reshape(-1,1),sigma.reshape(1,-1))
    affinity_X = np.exp(- np.divide((dist** 2),Sigma_mat ))
    #------------------------------------
    # affinity_X = affinity_X*(affinity_X>.001).astype('int')
    #------------------------------------
    clustering = SpectralClustering(affinity='precomputed',n_clusters=K, assign_labels="kmeans")
    labels= clustering.fit_predict(affinity_X)+1
    results_df ,_ = result_intprtn2(labels, y ,X ,'one')       
    results[j , :] = results_df.values
    j = j+1
'''
