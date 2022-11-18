# -*- coding: utf-8 -*-
"""
Created on Wed may 26 09:45:04 2021

@author: elahe
"""
import numpy as np
from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from jensenshannon import jensenshannon
import math
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from scipy.spatial import distance


def caculate_affinity(dist , type_affinity='local' , p = 2 , S = 7):
    #adaptiveDenisity: Spectrum: fast density-aware spectral clustering for
    #single and multi-omic data
    #local:  Self-tuning spectral clustering.
    
    N= dist.shape[0]
    sigma_all = np.sort(dist ,axis = 1)
    sigma_id_all = np.argsort(dist ,axis = 1)
    sigma = np.zeros([N,1])
    for ii in range(N):
        id_nonzero = next((i for i, x in enumerate(sigma_all[ii,:]) if x), None)
        sigma[ii] = sigma_all[ii, id_nonzero+p]#4
    Sigma_mat = np.matmul(sigma.reshape(-1,1),sigma.reshape(1,-1))

    if(type_affinity=='local'):   
        affinity_X = np.exp(-np.divide((dist** 2),Sigma_mat ))

    else:
        range_S_i = np.zeros([N,S])
        for ii in range(N):
            id_nonzero = next((i for i, x in enumerate(sigma_all[ii,:]) if x), None)
            range_S_i[ii , :] = sigma_id_all[ii, id_nonzero:id_nonzero+S]
        
        CNN = np.ones([N,N])
        for ii in range(N):
            for jj in range(ii+1 , N):
                xy,_, _ = np.intersect1d(range_S_i[ii , :], range_S_i[jj , :], return_indices=True)
                CNN[ii , jj] = len(xy)+1
                CNN[jj , ii] = CNN[ii , jj]
                
        Sigma_mat_CNN = Sigma_mat*CNN
        affinity_X = np.exp(- np.divide((dist** 2),Sigma_mat_CNN ))
        
    return affinity_X

def calculate_R_MI(X , no_f_cont):
    no_f=  X.shape[1]
    R = np.zeros([no_f ,no_f ])
    
    for r in range(no_f):
        x_r = X[:,r].reshape(-1,1)
        for s in range(r, no_f):
            x_s = X[:,s].reshape(-1,1)
            if(s<no_f_cont and r< no_f_cont):
                R[r , s]= mutual_info_regression(x_r, x_s.ravel(),n_neighbors=5 , random_state=0)
            elif(r<no_f_cont and s>=no_f_cont):
                R[r , s]= mutual_info_classif(x_r, x_s.ravel(), n_neighbors=5, random_state=0)
            elif(r>=no_f_cont and s>=no_f_cont):
                R[r , s]= mutual_info_score(x_r.ravel(), x_s.ravel())
                
            # max_cat_rs = int(np.max([np.max(x_s) , np.max(x_r)]))
            # R[r , s]/=np.log(max_cat_rs)
            R[s , r] = R[r , s]


    diag_r = np.diag(R)
    # for r in range(no_f):
    #     for rr in range(no_f):
    #         R[r , rr] =2*R[r , rr]/(diag_r[r] +diag_r[rr])

    for r in range(no_f):
        R[r , :] /=diag_r[r]    
    return R

def distace_categorical(X, no_f_cont , R , DM):#nominal 
    N = X.shape[0]
    no_f = X.shape[1]
    n_sampling = 1001
    no_f_disc = no_f - no_f_cont
    #-----------------------------------------------------------------
    #                       Initialization
    #-----------------------------------------------------------------
    max_cats = np.zeros([no_f_disc , 1])
    phi_rth = []
    for r in range(no_f_disc):
        max_cats[r] = int(np.max(X[: , r+no_f_cont]))
        phi_rth.append(np.zeros([int(max_cats[r]) , int(max_cats[r])]))
    
    
    dist_r_s_th = defaultdict(list)
    dist_rth = defaultdict(list)
    D_rth = defaultdict(list)
    
    for r in range(no_f_disc):
        max_cat_r = int(max_cats[r])
        dist_r_s_th[r]= defaultdict(list)
        dist_rth[r] = np.zeros([max_cat_r,max_cat_r ])
        D_rth[r] = np.zeros([max_cat_r,max_cat_r ])
        for s in range(no_f_cont):
            dist_r_s_th[r][s] = np.zeros([max_cat_r,max_cat_r ])
    #-----------------------------------------------------------------
    #                       Distane Calculation
    #-----------------------------------------------------------------
    
    #-----------------------------------------------------------------
    #          1. calculation of distance for categorical features:
    #-----------------------------------------------------------------                       
    S_As = np.log2(max_cats) 
    
    for r in range(no_f):
        if(r>=no_f_cont):
            max_cat_r = int(np.max(X[: , r]))
            if(DM !='DM1' and DM !='DM2'):
                for s in range(no_f_cont):
                    if(R[r , s]>0):                   
                        KDEs = np.zeros([n_sampling , max_cat_r]) 
                        x_s= X[: , s].reshape(-1, 1)
                        a1 = np.min(x_s)
                        a2 = np.max(x_s)
                        sm = np.std(x_s)
                        ni = len(x_s)                
                        h0 = 1.06*sm*ni**(-1/5)
                        X_plot = np.linspace(a1, a2, n_sampling).reshape(-1, 1)           
                        for i in range(max_cat_r):
                            logic_ri = X[: , r]==i+1
                            p_r_i = np.sum(logic_ri)/N
                            X_s_ri = X[logic_ri , s].reshape(-1, 1)*p_r_i  ### *p_r_i               
                            sm_sr = np.std(X_s_ri)
                            ni = len(X_s_ri)
        
                            if(sm_sr==0):
                                sm_sr = .01
                            h0 =1.06*sm_sr*ni**(-1/5)#.9                      
                            kde = KernelDensity(kernel='gaussian' , bandwidth=h0).fit(X_s_ri)
                            log_dens = kde.score_samples(X_plot)
                            KDEs[ : , i] = np.exp(log_dens) #columns shows diffrent categories                     
    
                        for t in range(max_cat_r):
                            for h in range(t , max_cat_r): 
                                jsd = jensenshannon(KDEs[ : , h],KDEs[ : , t] , 2)
                                if jsd>1:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  0#1
                                else:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  jsd
                            if(sm_sr==0):
                                sm_sr=.01
                            h0 =1.06*sm_sr*ni**(-1/5)#.9
                            #KDE: 'tophat','epanechnikov','gaussian'                        
                            kde = KernelDensity(kernel='gaussian' , bandwidth=h0).fit(X_s_ri)
                            log_dens = kde.score_samples(X_plot)
                            KDEs[ : , i] = np.exp(log_dens) #columns shows diffrent categories                     
        
                        for t in range(max_cat_r):
                            for h in range(t , max_cat_r): 
                                jsd = jensenshannon(KDEs[ : , h],KDEs[ : , t] , 2)    
                                if jsd>1:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  0
                                elif(math.isnan(jsd)):
                                    dist_r_s_th[r-no_f_cont][s][t , h] = 0
                                else:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  jsd
                        
            for t in range(max_cat_r):
                for h in range(t+1,max_cat_r):
                    phi_rth_s = np.zeros([no_f_disc,1])
                    for s in range(no_f_cont,no_f): 
                        if(R[r , s]>0):
                            max_cat_s = int(max_cats[s-no_f_cont])
                            P_rth_ss = np.zeros([max_cat_s,1])
                            E_rth_ss = np.zeros([max_cat_s,1])
                            for ss in range(max_cat_s):
                                P_rth_ss[ss] = np.sum(np.logical_and(X[: , r]==t+1 ,X[: , s]==ss+1 ))/N + np.sum(np.logical_and(X[: , r]==h+1 ,X[: , s]==ss+1 ))/N
                                if(P_rth_ss[ss]):
                                    E_rth_ss[ss] = -P_rth_ss[ss]*np.log2(P_rth_ss[ss])
                                           
                            phi_rth_s[s-no_f_cont] = np.sum(E_rth_ss)/S_As[s-no_f_cont]
                        
                            R_E_rth_s = phi_rth_s*R[r , no_f_cont:].reshape(-1 ,1)    
                            phi_rth[r-no_f_cont][t , h] = np.sum(R_E_rth_s)#/no_f_disc 
                            phi_rth[r-no_f_cont][h , t] = phi_rth[r-no_f_cont][t , h]
                        
    #-----------------------------------------------------------------------------                    
    # Combination of distance of categorical part based on discite and continues relevencis                
    #-----------------------------------------------------------------------------
    for r in range(no_f_disc):
        max_cat_r = int(np.max(X[: , r+no_f_cont] , axis = 0))
        for t in range(max_cat_r):
            for h in range(t , max_cat_r): 
                d_sr_th = 0
                for s in range(no_f_cont):
                    d_sr_th += dist_r_s_th[r][s][t , h]*R[r , s]#
                    
                dist_rth[r][t , h] = d_sr_th#/no_f_cont
                dist_rth[r][h , t] = d_sr_th
                
                if(DM == 'DM1' or DM =='DM2'):
                    D_rth[r][t , h] =(phi_rth[r][t , h])/no_f_disc##
                else:
                    D_rth[r][t , h] =(phi_rth[r][t , h]+ dist_rth[r][t , h])/no_f##
                
                D_rth[r][h , t] = D_rth[r][t , h]       
    return D_rth

def distace_categorical_ordinal_nominal(X, no_f_cont ,no_f_ord, R, DM):
    N = X.shape[0]
    no_f = X.shape[1]
    n_sampling = 1001
    no_f_disc = no_f - no_f_cont

    #-----------------------------------------------------------------
    #                       Initialization
    #-----------------------------------------------------------------
    max_cats = np.zeros([no_f_disc , 1])
    phi_rth = []
    for r in range(no_f_disc):
        max_cats[r] = int(np.max(X[: , r+no_f_cont]))
        phi_rth.append(np.zeros([int(max_cats[r]) , int(max_cats[r])]))
    
    
    dist_r_s_th = defaultdict(list)
    dist_rth = defaultdict(list)
    D_rth = defaultdict(list)
    
    for r in range(no_f_disc):
        max_cat_r = int(max_cats[r])
        dist_r_s_th[r]= defaultdict(list)
        dist_rth[r] = np.zeros([max_cat_r,max_cat_r ])
        D_rth[r] = np.zeros([max_cat_r,max_cat_r ])
        for s in range(no_f_cont):
            dist_r_s_th[r][s] = np.zeros([max_cat_r,max_cat_r ])
    #-----------------------------------------------------------------
    #                       Distane Calculation
    #-----------------------------------------------------------------
    
    #-----------------------------------------------------------------
    #          1. calculation of distance for categorical features:
    #-----------------------------------------------------------------                       
    S_As = np.log2(max_cats) 
    
    for r in range(no_f):
    #-----------------------------------------------------------------
    #                r is dicrite and s is continues
    #-----------------------------------------------------------------
        if(r>=no_f_cont):
            max_cat_r = int(np.max(X[: , r]))
            if(DM !='DM1' and DM !='DM2'):
                for s in range(no_f_cont):
                    if(R[r , s]>0):                   
                        KDEs = np.zeros([n_sampling , max_cat_r]) 
                        x_s= X[: , s].reshape(-1, 1)
                        a1 = np.min(x_s)
                        a2 = np.max(x_s)
                        sm = np.std(x_s)
                        ni = len(x_s)
                        h0 = 1.06*sm*ni**(-1/5)
                        X_plot = np.linspace(a1, a2, n_sampling).reshape(-1, 1)
                        for i in range(max_cat_r):
                            logic_ri = X[: , r]==i+1
                            p_r_i = np.sum(logic_ri)/N
                            X_s_ri = X[logic_ri , s].reshape(-1, 1)*p_r_i  ### *p_r_i               
                            sm_sr = np.std(X_s_ri)
                            ni = len(X_s_ri)
                            if(ni==0):
                                KDEs[ : , i] = np.zeros([n_sampling,1]).ravel()
                            else: 
                                # print(r , sm_sr , ni)
                                if(sm_sr==0):
                                    sm_sr=.01
                                h0 =1.06*sm_sr*ni**(-1/5)#.9
                                #KDE: 'tophat','epanechnikov','gaussian'                        
                                kde = KernelDensity(kernel='gaussian' , bandwidth=h0).fit(X_s_ri)
                                log_dens = kde.score_samples(X_plot)
                                KDEs[ : , i] = np.exp(log_dens) #columns shows diffrent categories                     
        
                        for t in range(max_cat_r):
                            for h in range(t , max_cat_r): 
                                jsd = jensenshannon(KDEs[ : , h],KDEs[ : , t] , 2)
                                # print(jsd)
                                if jsd>1:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  0
                                elif(math.isnan(jsd)):
                                    dist_r_s_th[r-no_f_cont][s][t , h] = 0
                                else:
                                    dist_r_s_th[r-no_f_cont][s][t , h] =  jsd
                                # print(dist_r_s_th[r-no_f_cont][s][t , h])
    #-----------------------------------------------------------------
    #                r is ordinal and s is discrite
    #-----------------------------------------------------------------
        if(r<no_f_ord+no_f_cont and r>=no_f_cont):# if r is ordinal
            for t in range(max_cat_r):
                for h in range(t+1,max_cat_r):
                    phi_rth_s = np.zeros([no_f_disc,1])
                    for s in range(no_f_cont,no_f): 
                        min_g = int(min(h,t))
                        max_g = int(max(h,t))
                        E_rth_g = np.zeros([max_g-min_g,1])
    
                        for g in range( min_g, max_g ):
                            
                            max_cat_s = int(max_cats[s-no_f_cont])
                            P_rth_ss = np.zeros([max_cat_s,1])
                            E_rth_ss = np.zeros([max_cat_s,1])
    
                            for ss in range(max_cat_s):
                                P_rth_ss[ss] = np.sum(np.logical_and(X[: , r]==g+1 ,X[: , s]==ss+1 ))/N + np.sum(np.logical_and(X[: , r]==g+2 ,X[: , s]==ss+1 ))/N
                                if(P_rth_ss[ss]):
                                    E_rth_ss[ss] = -P_rth_ss[ss]*np.log2(P_rth_ss[ss])
                                
                            E_rth_g[g-min_g] =  np.sum(E_rth_ss)
                            
                        phi_rth_s[s-no_f_cont] = np.sum(E_rth_g)/S_As[s-no_f_cont]
                        
                    R_E_rth_s = R[r , no_f_cont:].reshape(-1 ,1)* phi_rth_s                   
                    phi_rth[r-no_f_cont][t , h] = np.sum(R_E_rth_s)
                    phi_rth[r-no_f_cont][h , t] = phi_rth[r-no_f_cont][t , h]                   
    ##------------------------------nominal---------------------------
    #                r is nominal and s is discrite
    #-----------------------------------------------------------------
        elif(r>=no_f_ord+no_f_cont):# if r is nominal 
            for t in range(max_cat_r):
                for h in range(t+1,max_cat_r):
                    phi_rth_s = np.zeros([no_f_disc,1])
                    for s in range(no_f_cont,no_f): 
                        max_cat_s = int(max_cats[s-no_f_cont])
                        P_rth_ss = np.zeros([max_cat_s,1])
                        E_rth_ss = np.zeros([max_cat_s,1])
                        for ss in range(max_cat_s):
                            P_rth_ss[ss] = np.sum(np.logical_and(X[: , r]==t+1 ,X[: , s]==ss+1 ))/N + np.sum(np.logical_and(X[: , r]==h+1 ,X[: , s]==ss+1 ))/N
                            if(P_rth_ss[ss]):
                                E_rth_ss[ss] = -P_rth_ss[ss]*np.log2(P_rth_ss[ss])
                            
                        phi_rth_s[s-no_f_cont] = np.sum(E_rth_ss)/S_As[s-no_f_cont]
        
                    R_E_rth_s = R[r , no_f_cont:].reshape(-1 ,1)* phi_rth_s   
                    phi_rth[r-no_f_cont][t , h] = np.sum(R_E_rth_s)
                    phi_rth[r-no_f_cont][h , t] = phi_rth[r-no_f_cont][t , h]
    
    #------------------------------------------------------------------------
    #       Combination of distance of categorical part based on discite 
    #                     and continues relevencis                
    #------------------------------------------------------------------------
    
    for r in range(no_f_disc):
        max_cat_r = int(np.max(X[: , r+no_f_cont] , axis = 0))
        for t in range(max_cat_r):
            for h in range(t , max_cat_r): 
                d_sr_th = 0
                for s in range(no_f_cont):
                    d_sr_th += dist_r_s_th[r][s][t , h]*R[r , s]#
                    
                dist_rth[r][t , h] = d_sr_th#/no_f_cont
                dist_rth[r][h , t] = d_sr_th
                
                if(DM == 'DM1' or DM =='DM2'):
                    D_rth[r][t , h] =(phi_rth[r][t , h])/no_f_disc##
                else:
                    D_rth[r][t , h] =(phi_rth[r][t , h]+ dist_rth[r][t , h])/no_f##
                D_rth[r][h , t] = D_rth[r][t , h]    
    return D_rth

def ditance_dependency_mixed_matrix(XX ,no_f_cont , no_f_ord , DM , matlab_call='False'):
    X = XX.copy()
    N = X.shape[0]
    no_f = X.shape[1]
    sc =  MinMaxScaler()#Normalizer()#StandardScaler()
    rep_fea = []
    for i in range(no_f):
        if(len(np.unique(X[: , i]))==1):
          rep_fea.append(i) 

    # for j in range(len(rep_fea)):
    #     if(rep_fea[j]>):
            
    X = np.delete(X, rep_fea , axis = 1) 
    no_f=no_f-len(rep_fea)  
    no_f_ord=no_f_ord-len(rep_fea)  

    no_f_disc = no_f - no_f_cont
    
    if(no_f_cont):
        X[:,:no_f_cont ] = sc.fit_transform(X[:,:no_f_cont ])

    R = calculate_R_MI(X , no_f_cont)

    if(DM =='DM0'):
        D_rth = 0
    elif(DM =='DM1'):
        D_rth = distace_categorical(X, no_f_cont, R , DM)
    else:
        D_rth = distace_categorical_ordinal_nominal(X, no_f_cont, no_f_ord, R , DM)


    #-----------------------------------------------------------------------------
    #       2 .calculation of pairwise distance based on categorical part between N samples
    #-----------------------------------------------------------------------------
    dist_discrite = np.zeros([N ,N])
    dist_ij_r = np.zeros([no_f_disc, 1])
    # w_cont = np.sum(R[:no_f_cont , :] , axis =1 )
    
    for i in range(N):
        for j in range(i , N):
            if(DM=='DM0'):
                dist_discrite[i,j] = (no_f_disc/(no_f_disc+1))*distance.hamming(X[i ,:], X[j,:])
            else:
                for r in range(no_f_cont , no_f):
                    x_i = int(X[i ,r]-1)
                    x_j = int(X[j ,r]-1)
                    dist_ij_r[r-no_f_cont] = D_rth[r-no_f_cont][x_i , x_j]
                
                dist_discrite[i,j] = np.sum(dist_ij_r)
            dist_discrite[j,i] = dist_discrite[i,j]
        

    #-----------------------------------------------------------------
    #          3. calculation of distance for Continues features:
    #-----------------------------------------------------------------                       
    if(DM=='DM4'):#orinary mahalonobis        
        if(no_f_cont==1): 
            VI = np.cov(X[: , :no_f_cont].T)**(-1)
        else:
            VI = np.linalg.inv(np.cov(X[: , :no_f_cont].T))# and .2 for wine
        MD = cdist(X[: , :no_f_cont],X[: , :no_f_cont], 'mahalanobis' , VI=VI)

    elif(DM=='DM5'):#my modified mahalonobis       
        if(no_f_cont==1): 
            VI = np.cov(X[: , :no_f_cont].T)**(-1)*R[:no_f_cont , :no_f_cont ]
        else: 
            VI = np.multiply(np.linalg.inv(np.cov(X[: , :no_f_cont].T)) ,R[:no_f_cont , :no_f_cont ]) # and .2 for wine
            # VI = (R[:no_f_cont , :no_f_cont ]+R.T)/2 # and .2 for wine

        MD = cdist(X[: , :no_f_cont],X[: , :no_f_cont], 'mahalanobis' , VI=VI)

    else:#euclidean distance
        MD = cdist(X[: , :no_f_cont],X[: , :no_f_cont], 'euclidean' )
    dist_continues = MD     
       
    
    dist =  (1/(no_f_disc+1))*dist_continues + (no_f_disc/(no_f_disc+1))*dist_discrite
    # dist =  ((no_f-no_f_disc)/(no_f))*dist_continues + (no_f_disc/(no_f))*dist_discrite

    if(matlab_call == True):
        with open('dist_matlab.txt') as f:
            for line in dist:
                np.savetxt(f, line, fmt='%.2f')
    return dist


