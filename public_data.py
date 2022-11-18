# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:12:12 2021

@author: elahe
"""
import pandas as pd
import numpy as np
def public_data_calling(name='celev_heart' , type_data='mixed' , 
                        no_fgid='one' , no_fgid_feature = 30):

    #---------------------------Mixed Data------------------------------------
    if(type_data=='mixed'):
        if(name=='adault2_dep'):
            Data = pd.read_csv('Data/adault2_dep.csv')
            Data = Data.values
            no_f_cont = 6
            no_f_ord = 0
            K = 2
        
        elif(name == 'credit'):
            Data = pd.read_csv('Data/credit_approval3.csv')
            Data = Data.values
            no_f_cont = 6
            no_f_ord = 1
            K = 2
            
        elif(name=='australian'):
            Data = pd.read_csv('Data//australian.csv')
            Data = Data.values
            no_f_cont = 6
            no_f_ord = 2
            K = 2
        
        elif(name=='dermatology'):
            Data = pd.read_csv('Data//dermatology.csv')
            Data = Data.values
            no_f_cont =1#33
            no_f_ord = 32#0#
            K = 6
            
        elif(name=='statlog_heart'):
            Data = pd.read_csv('Data/statlog_heart.csv')
            Data = Data.values
            no_f_cont =6
            no_f_ord = 2
            K = 2
            
        elif(name=='german'):    
            Data = pd.read_csv('Data/german.csv')
            Data = Data.values
            no_f_cont = 7
            K = 2
            no_f_ord = 2
        
        elif(name=='kaggle_heart'):    
            Data = pd.read_csv('Data/heart.csv')
            Data = Data.values
            no_f_cont =6
            no_f_ord = 2
            K =2
            
        elif(name=='celev_heart'):    
            Data = np.loadtxt('Data/heart3.txt')#prl3 data
            # Data = Data.values
            no_f_cont =6
            no_f_ord = 2
            K =2
            
        elif(name =='LE'): 
            Data = np.loadtxt('Data/LE.txt')#ordinal data
            Data = Data.astype('int')
            no_f_ord = 4
            no_f_cont =0
            K = 5
            
        elif(name=='car'):    
            Data = pd.read_csv('Data/car.csv') #ordinal
            Data = Data.values
            no_f_cont = 0
            no_f_ord = 6
            K = 4
            
        elif(name=='breast'):    
            Data = pd.read_csv('Data/breast.csv')#ordinal
            Data = Data.values
            no_f_cont = 0
            no_f_ord = 9
            K = 2
            
        elif(name=='vote'):    
            Data = pd.read_csv('Data/vote.csv')#nominal
            Data = Data.values
            no_f_cont = 0
            no_f_ord = 0
            K = 2
    
        elif(name=='soybean_small'):
            Data = pd.read_csv('Data/soybean_samll.csv')#nominal
            Data = Data.values
            no_f_cont = 0
            no_f_ord = 0
            K = 4
               
        elif(name =='HR'):
            Data = np.loadtxt('Data/HR.txt')#ordinal+nominal
            Data = Data.astype('int')
            no_f_ord = 2
            no_f_cont = 0
            K = 3
    
        elif(name =='MM'):
            Data = np.loadtxt('Data/MM.txt')#ordinal+nominal
            Data = Data.astype('int')
            no_f_ord = 2
            no_f_cont =0
            K = 2

        elif(name =='LG'):
            Data = np.loadtxt('Data/LG.txt')#ordinal+nominal
            Data = Data.astype('int')
            no_f_ord = 3
            no_f_cont =0
            K = 4
                    
        elif(name =='ion'):
            Data = pd.read_csv('Data/ionosphere.csv')   
            Data = Data.drop(['a02'], axis =1)
            Data = Data.values
            no_f_ord = 0
            no_f_cont =33
            K = 2
            
        elif(name =='sonar'):
            Data = pd.read_csv('Data/sonar.csv')   
            Data = Data.values
            no_f_ord = 0
            no_f_cont =60
            K = 2
            
        elif(name =='yeast'):
            Data = pd.read_csv('Data/yeast2.csv')   
            Data = Data.values
            no_f_ord = 0
            no_f_cont =8
            K = 10
        elif(name =='ecoli'):
            Data = pd.read_csv('Data/ecoli.csv')   
            Data = Data.drop(['\tchg'], axis =1)
            Data = Data.values
            no_f_ord = 0
            no_f_cont = 6
            K = 8  
            
        elif(name =='iris'):
            Data = np.loadtxt('Data/iris.txt')#numerical
            no_f_ord = 0
            no_f_cont =Data.shape[1]-1
            K = int(np.max(Data[:,-1]))

        elif(name =='wine'):
            Data = np.loadtxt('Data/wine.txt')#numerical
            no_f_ord = 0
            no_f_cont =Data.shape[1]-1
            K = int(np.max(Data[:,-1]))
   
        X = Data[: , :-1]
        y = Data[: , -1]
        
        N = X.shape[0]
        no_f = X.shape[1] 

    
    return X , y, K , N ,no_f, no_f_cont , no_f_ord