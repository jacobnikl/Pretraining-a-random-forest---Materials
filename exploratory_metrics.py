#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:31:50 2024

@author: niklasjacobs
"""

import pandas as pd
import numpy as np
import sklearn as sklearn
import warnings
import matplotlib.pyplot as plt
import math
import miceforest as mf
import datetime as dt
import scipy
import random
import pingouin as pg
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import t
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTENC
from collections import Counter
from scipy.stats import binned_statistic

warnings.filterwarnings("ignore")

execution_start = dt.datetime.now()

pd.set_option('display.max_rows', None)

iterations = 100
est_imput = 10
it_impute = 10

def load_data():
    """Import patient Data and string containing used variables and YBOCS post value"""
    
    global PATH_WORKINGDIRECTORY, OPTIONS_OVERALL

    data_import_path = ""
    data_import = pd.read_excel(data_import_path)
    
    string_import_path = ""
    string_import = read_csv(string_import_path, sep=";", header=0)
    
    string_import = string_import.drop(string_import.columns[0], axis=1)
    string_import = string_import.drop(string_import.columns[502:1000], axis=1)

    common_columns = string_import.columns
    data_red = data_import[common_columns]
    
    ## reassign dtypes of columns

    "replace number coded missings + typo with NaN"
    
    data_red = data_red.replace([777, 888, 999], np.nan)
    data_red['Beginn_Zwang'] = data_red['Beginn_Zwang'].clip(upper=100).replace(100, np.nan)
    data_red['Beginn_Zwang_DSM'] = data_red['Beginn_Zwang_DSM'].clip(upper=100).replace(100, np.nan)
    data_red.at[135,'MADSUMT0'] = np.nan

    # drop columns with >= 25% missings
    
    missing_percent = (data_red.isnull().sum() / len(data_red)) * 100
    columns_to_drop = missing_percent[missing_percent >= 25].index
    data_red = data_red.drop(columns=columns_to_drop)
    
    # drop rows with >= 25% missings

    missing_percent = (data_red.isnull().sum(axis=1) / data_red.shape[1]) * 100
    rows_to_drop = missing_percent[missing_percent >= 25].index
    data_red = data_red.drop(index=rows_to_drop)
    

    return data_red



def binarise(data, pretest, posttest): 
    """"Define outcome-variable using the RCI"""
    
    rel=0.866
    sd_pre = data.iloc[:,pretest].std()
    SEM = sd_pre * math.sqrt(1-rel)
    SEM_sq = SEM**2
    SE_diff = math.sqrt(2 * (SEM_sq))
    
    for i in range(data.shape[0]):
        data.at[i, 'RCI'] = (data.iat[i, posttest] - data.iat[i, pretest]) / SE_diff
        
    RCI_c_num = data.columns.get_loc('RCI')

    for i in range(data.shape[0]):
        if data.iat[i,RCI_c_num ] <= -1.96:
            data.at[i, 'resp'] = 1
        else:
            data.at[i, 'resp'] = 0
        
    return data


def do_traintestsplit(X,y,i):
    """Conduct a single train-test split with 20% of data in test """
    
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, test_size=0.2, random_state=i)
    
    return X_train, X_test, y_train, y_test

def threshold_predictions(y_prob, threshold):
    "obtaining a classifiers prediction for a specific threshold value to obtain true and false positive rates"
    return (y_prob >= threshold).astype(int)
            
def do_predictions(X_train, X_test, y_train, y_test, i, params, SMOTE = False):
    """Train standard Random Forest classifier on train split and get predictions on test data"""
    
    if SMOTE == True:
    
        # apply SMOTE
        
        vector = ["BSIMITT0", "BSIM00T0", "BSIM01T0", "BSIM02T0", "BSIM03T0", "BSIM04T0", "BSIM05T0", "BSIM06T0", "BSIM07T0", "BSIM08T0", "BSIM09T0", "YBOSUMT0"]
        cat_columns_smote = [col for col in X_train.columns if col not in vector]
        sm = SMOTENC(random_state=i, categorical_features=cat_columns_smote)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        print('Resampled dataset shape %s' % Counter(y_train))
            
    print('Resampled dataset shape %s' % Counter(y_train))
    
    clf = RandomForestClassifier(n_estimators=200, random_state=i)
    clf.set_params(**params)
    clf = clf.fit(X_train, y_train)
    
    accuracy_train = balanced_accuracy_score(y_train, clf.predict(X_train))
    accuracy_s = balanced_accuracy_score(y_test, clf.predict(X_test))
    
    sensitivity, specificity = get_sens_spec(y_true=y_test, y_pred=clf.predict(X_test))

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy_30 = balanced_accuracy_score(y_test,threshold_predictions(y_pred_prob, 0.3))
    accuracy_70 = balanced_accuracy_score(y_test,threshold_predictions(y_pred_prob, 0.7))

    tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()

    return accuracy_s, accuracy_train, y_pred_prob, accuracy_30, accuracy_70, tn, fp, fn, tp

        
def get_sens_spec (y_true, y_pred):
    "extract sensitivity and specificity from test labels and predicted labels"
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    return sensitivity, specificity


def rep(value, n):
    "function to paste a value (for instance mean) n times, used for weighted coefficients in simcor"
    reps = []
    for _ in range(n):
        reps.append(value)
    return reps

def ttc(cor, sd=None, var=None):
    "turn a correlation into a covariance using either its sd or variance"
    
    if sd is not None and len(sd) > 0:
        cov = cor * sd[0] * sd[1]
    else:
        cov = cor * np.sqrt(var[0]) * np.sqrt(var[1])
    return cov

def SIMCOR (findings, cor_file,number_of_Variables_1, number_of_Variables_2, RL, X_train, y_train, i_seed):
    "extract values found in the literature from excels, then simulate data using these values"

    ## define lists to store the correlations
    
    cor_YB = []
    cor_YA = []
    cor_YO = []
    cor_YG = []
    cor_YOC = []
    cor_BA = []
    cor_BO = []
    cor_BG = []
    cor_BOC = []
    cor_AO = []
    cor_AG = []
    cor_AOC = []
    cor_OG = []
    cor_OOC = []
    cor_GOC = []

    
    ## define a variable that contains every variable name only once
    
    unique_entries = cor_file['Variable '].unique()
    
    
    for i in number_of_Variables_1:
        
        ## which variable matches the unique entry accessed by i
        
        VARIABLE = cor_file[cor_file['Variable '] == unique_entries[i]]
        
        ## extract the covariance from the current row
        
        for i in range(len(VARIABLE)):
            
            current = VARIABLE.iloc[[i]]
            current = current.T
            
            ## use rep to paste value as often as its n
            
            add_cor = rep(current.iat[1,0],current.iat[2,0])
            
            ## add the correlation to the list it belongs to (by name)
            
            if current.iloc[0,0] == "YB":
                
                YB_cor_create = []
                
                YB_cor_create.append(add_cor)
                
                for sublist in YB_cor_create:
                    cor_YB.extend(sublist)
                
            else:
                                
                if current.iloc[0,0] == "YA":
                        
                    YA_cor_create = []
                        
                    YA_cor_create.append(add_cor)
                        
                    for sublist in YA_cor_create:
                        cor_YA.extend(sublist)
                
                else: 
                    
                    if current.iloc[0,0] == "YO":
                            
                        YO_cor_create = []
                            
                        YO_cor_create.append(add_cor)
                            
                        for sublist in YO_cor_create:
                            cor_YO.extend(sublist)
                            
                    else: 
                        if current.iloc[0,0] == "YG":
                                
                            YG_cor_create = []
                                
                            YG_cor_create.append(add_cor)
                                
                            for sublist in YG_cor_create:
                                cor_YG.extend(sublist)
                                
                        else: 
                            if current.iloc[0,0] == "YOC":
                                    
                                YOC_cor_create = []
                                    
                                YOC_cor_create.append(add_cor)
                                    
                                for sublist in YOC_cor_create:
                                    cor_YOC.extend(sublist)
                            else: 
                                
                                if current.iloc[0,0] == "BA":
                                        
                                    BA_cor_create = []
                                        
                                    BA_cor_create.append(add_cor)
                                        
                                    for sublist in BA_cor_create:
                                        cor_BA.extend(sublist)
                                    
                                else: 
                                    
                                    if current.iloc[0,0] == "BO":
                                            
                                        BO_cor_create = []
                                            
                                        BO_cor_create.append(add_cor)
                                            
                                        for sublist in BO_cor_create:
                                            cor_BO.extend(sublist)
                                            
                                    else: 
                                        if current.iloc[0,0] == "BG":
                                                
                                            BG_cor_create = []
                                                
                                            BG_cor_create.append(add_cor)
                                                
                                            for sublist in BG_cor_create:
                                                cor_BG.extend(sublist)
                                                
                                        else: 
                                            if current.iloc[0,0] == "BOC":
                                                    
                                                BOC_cor_create = []
                                                    
                                                BOC_cor_create.append(add_cor)
                                                    
                                                for sublist in BOC_cor_create:
                                                    cor_BOC.extend(sublist)
                                            
                                            else: 
                                                if current.iloc[0,0] == "AO":
                                                        
                                                    AO_cor_create = []
                                                        
                                                    AO_cor_create.append(add_cor)
                                                        
                                                    for sublist in AO_cor_create:
                                                        cor_AO.extend(sublist)
                                                        
                                                else: 
                                                    if current.iloc[0,0] == "AG":
                                                            
                                                        AG_cor_create = []
                                                            
                                                        AG_cor_create.append(add_cor)
                                                            
                                                        for sublist in AG_cor_create:
                                                            cor_AG.extend(sublist)
                                                            
                                                    else: 
                                                        if current.iloc[0,0] == "AOC":
                                                                
                                                            AOC_cor_create = []
                                                                
                                                            AOC_cor_create.append(add_cor)
                                                                
                                                            for sublist in AOC_cor_create:
                                                                cor_AOC.extend(sublist)
                                                        
                                                        else: 
                                                            if current.iloc[0,0] == "OG":
                                                                    
                                                                OG_cor_create = []
                                                                    
                                                                OG_cor_create.append(add_cor)
                                                                    
                                                                for sublist in OG_cor_create:
                                                                    cor_OG.extend(sublist)
                                                                    
                                                            else: 
                                                                if current.iloc[0,0] == "OOC":
                                                                        
                                                                    OOC_cor_create = []
                                                                        
                                                                    OOC_cor_create.append(add_cor)
                                                                        
                                                                    for sublist in OOC_cor_create:
                                                                        cor_OOC.extend(sublist)
                                                                        
                                                                else: 
                                                                    if current.iloc[0,0] == "GOC":
                                                                             
                                                                        GOC_cor_create = []
                                                                             
                                                                        GOC_cor_create.append(add_cor)
                                                                             
                                                                        for sublist in GOC_cor_create:
                                                                             cor_GOC.extend(sublist)
                                                                
                                
    
    ## get the mean value of the list -> correlations weighted by n                                                

    cor_YB = sum(cor_YB)/len(cor_YB)
    cor_YA = sum(cor_YA)/len(cor_YA)
    cor_YO = sum(cor_YO)/len(cor_YO)
    cor_YG = sum(cor_YG)/len(cor_YG)
    cor_YOC = sum(cor_YOC)/len(cor_YOC)
    cor_BA = sum(cor_BA)/len(cor_BA)
    cor_BO = sum(cor_BO)/len(cor_BO)
    cor_BG = sum(cor_BG)/len(cor_BG)
    cor_BOC = sum(cor_BOC)/len(cor_BOC)
    cor_AO = sum(cor_AO)/len(cor_AO)
    cor_AG = sum(cor_AG)/len(cor_AG)
    cor_AOC = sum(cor_AOC)/len(cor_AOC)
    cor_OG = sum(cor_OG)/len(cor_OG)
    
    ## replace Onset/GAF + OCI covariances
    
    cor_OOC = X_train["OCISUMT0"].corr(X_train["Beginn_Zwang_DSM"], method = "pearson")
    cor_GOC = X_train["OCISUMT0"].corr(X_train["GAF"], method = "pearson")
    
    "now the same procedure for means and standard deviations"

    ## define lists to store means and standard deviations
    
    YBOCS_m_R = []
    YBOCS_m_NR = []
    YBOCS_s_R = []
    YBOCS_s_NR = []

    BDI_m_R =[]
    BDI_m_NR = []
    BDI_s_R = []
    BDI_s_NR = []
    
    Age_m_R =[]
    Age_m_NR = []
    Age_s_R = []
    Age_s_NR = []
    
    Onset_m_R =[]
    Onset_m_NR = []
    Onset_s_R = []
    Onset_s_NR = []
    
    GAF_m_R =[]
    GAF_m_NR = []
    GAF_s_R = []
    GAF_s_NR = []
        
    OCI_m_R =[]
    OCI_m_NR = []
    OCI_s_R = []
    OCI_s_NR = []
    
    ## define a variable that contains every variable name only once
    
    unique_entries = findings['Variable '].unique()

    ## extract variable that matches the unique entry accessed by i    

    for i in number_of_Variables_2:
        
        VARIABLE = findings[findings['Variable '] == unique_entries[i]]
        
        for i in range(len(VARIABLE)):
            
            ## extract means and standard deviations for the currently accessed row
            
            current = VARIABLE.iloc[[i]]
            current = current.T
            
            ## use rep to paste the value as often as its n
            
            add_m_R = rep(current.iat[1,0],current.iat[5,0])
            add_m_NR = rep(current.iat[2,0],current.iat[5,0])
            add_s_R = rep(current.iat[3,0],current.iat[5,0])
            add_s_NR = rep(current.iat[4,0],current.iat[5,0])
            
            ## add the extracted values to their respective lists by name

            if current.iloc[0,0] == "YBOCS":
                
                YBOCS_m_R_create = []
                YBOCS_s_R_create = []
                YBOCS_m_NR_create = []
                YBOCS_s_NR_create = []
                
                YBOCS_m_R_create.append(add_m_R)
                YBOCS_m_NR_create.append(add_m_NR)
                YBOCS_s_R_create.append(add_s_R)
                YBOCS_s_NR_create.append(add_s_NR)
                
                
                for sublist in YBOCS_m_R_create:
                    YBOCS_m_R.extend(sublist)
                for sublist in YBOCS_m_NR_create:
                    YBOCS_m_NR.extend(sublist)
                for sublist in YBOCS_s_R_create:
                    YBOCS_s_R.extend(sublist)
                for sublist in YBOCS_s_NR_create:
                    YBOCS_s_NR.extend(sublist)

            else: 
                 
                if current.iloc[0,0] == "BDI":
                    
                        
                    BDI_m_R_create = []
                    BDI_s_R_create = []
                    BDI_m_NR_create = []
                    BDI_s_NR_create = []
                        
                    BDI_m_R_create.append(add_m_R)
                    BDI_m_NR_create.append(add_m_NR)
                    BDI_s_R_create.append(add_s_R)
                    BDI_s_NR_create.append(add_s_NR)

                    
                    for sublist in BDI_m_R_create:
                        BDI_m_R.extend(sublist)
                    for sublist in BDI_m_NR_create:
                        BDI_m_NR.extend(sublist)
                    for sublist in BDI_s_R_create:
                        BDI_s_R.extend(sublist)
                    for sublist in BDI_s_NR_create:
                        BDI_s_NR.extend(sublist)
                    
                else: 
                    if current.iloc[0,0] == "Age":
                        
                            
                        Age_m_R_create = []
                        Age_s_R_create = []
                        Age_m_NR_create = []
                        Age_s_NR_create = []
                            
                        Age_m_R_create.append(add_m_R)
                        Age_m_NR_create.append(add_m_NR)
                        Age_s_R_create.append(add_s_R)
                        Age_s_NR_create.append(add_s_NR)

                        
                        for sublist in Age_m_R_create:
                            Age_m_R.extend(sublist)
                        for sublist in Age_m_NR_create:
                            Age_m_NR.extend(sublist)
                        for sublist in Age_s_R_create:
                            Age_s_R.extend(sublist)
                        for sublist in Age_s_NR_create:
                            Age_s_NR.extend(sublist)
                            
                    else: 
                        if current.iloc[0,0] == "Onset":
                            
                                
                            Onset_m_R_create = []
                            Onset_s_R_create = []
                            Onset_m_NR_create = []
                            Onset_s_NR_create = []
                                
                            Onset_m_R_create.append(add_m_R)
                            Onset_m_NR_create.append(add_m_NR)
                            Onset_s_R_create.append(add_s_R)
                            Onset_s_NR_create.append(add_s_NR)

                            
                            for sublist in Onset_m_R_create:
                                Onset_m_R.extend(sublist)
                            for sublist in Onset_m_NR_create:
                                Onset_m_NR.extend(sublist)
                            for sublist in Onset_s_R_create:
                                Onset_s_R.extend(sublist)
                            for sublist in Onset_s_NR_create:
                                Onset_s_NR.extend(sublist)
                            
                        else: 
                            if current.iloc[0,0] == "GAF":
                                
                                    
                                GAF_m_R_create = []
                                GAF_s_R_create = []
                                GAF_m_NR_create = []
                                GAF_s_NR_create = []
                                    
                                GAF_m_R_create.append(add_m_R)
                                GAF_m_NR_create.append(add_m_NR)
                                GAF_s_R_create.append(add_s_R)
                                GAF_s_NR_create.append(add_s_NR)

                                
                                for sublist in GAF_m_R_create:
                                    GAF_m_R.extend(sublist)
                                for sublist in GAF_m_NR_create:
                                    GAF_m_NR.extend(sublist)
                                for sublist in GAF_s_R_create:
                                    GAF_s_R.extend(sublist)
                                for sublist in GAF_s_NR_create:
                                    GAF_s_NR.extend(sublist)
                                
                            else: 
                                if current.iloc[0,0] == "OCI-R":
                                    
                                        
                                    OCI_m_R_create = []
                                    OCI_s_R_create = []
                                    OCI_m_NR_create = []
                                    OCI_s_NR_create = []
                                        
                                    OCI_m_R_create.append(add_m_R)
                                    OCI_m_NR_create.append(add_m_NR)
                                    OCI_s_R_create.append(add_s_R)
                                    OCI_s_NR_create.append(add_s_NR)

                                    
                                    for sublist in OCI_m_R_create:
                                        OCI_m_R.extend(sublist)
                                    for sublist in OCI_m_NR_create:
                                        OCI_m_NR.extend(sublist)
                                    for sublist in OCI_s_R_create:
                                        OCI_s_R.extend(sublist)
                                    for sublist in OCI_s_NR_create:
                                        OCI_s_NR.extend(sublist)
                                        

    ## get mean values of the lists -> means and sd weighted by n                  
     
    YBOCS_mean_R = sum(YBOCS_m_R)/len(YBOCS_m_R)
    YBOCS_mean_NR = sum(YBOCS_m_NR)/len(YBOCS_m_NR)
    YBOCS_sd_R = sum(YBOCS_s_R)/len(YBOCS_s_R)
    YBOCS_sd_NR = sum(YBOCS_s_NR)/len(YBOCS_s_NR)

    BDI_mean_R = sum(BDI_m_R)/len(BDI_m_R)
    BDI_mean_NR = sum(BDI_m_NR)/len(BDI_m_NR)
    BDI_sd_R = sum(BDI_s_R)/len(BDI_s_R)
    BDI_sd_NR = sum(BDI_s_NR)/len(BDI_s_NR)     
    
    Age_mean_R = sum(Age_m_R)/len(Age_m_R)
    Age_mean_NR = sum(Age_m_NR)/len(Age_m_NR)
    Age_sd_R = sum(Age_s_R)/len(Age_s_R)
    Age_sd_NR = sum(Age_s_NR)/len(Age_s_NR)    
    
    Onset_mean_R = sum(Onset_m_R)/len(Onset_m_R)
    Onset_mean_NR = sum(Onset_m_NR)/len(Onset_m_NR)
    Onset_sd_R = sum(Onset_s_R)/len(Onset_s_R)
    Onset_sd_NR = sum(Onset_s_NR)/len(Onset_s_NR)   
    
    GAF_mean_R = sum(GAF_m_R)/len(GAF_m_R)
    GAF_mean_NR = sum(GAF_m_NR)/len(GAF_m_NR)
    GAF_sd_R = sum(GAF_s_R)/len(GAF_s_R)
    GAF_sd_NR = sum(GAF_s_NR)/len(GAF_s_NR)   
    
    OCI_mean_R = sum(OCI_m_R)/len(OCI_m_R)
    OCI_mean_NR = sum(OCI_m_NR)/len(OCI_m_NR)
    OCI_sd_R = sum(OCI_s_R)/len(OCI_s_R)
    OCI_sd_NR = sum(OCI_s_NR)/len(OCI_s_NR)   
  
    
    ## create vector with the obtained means
    
    means_r = [YBOCS_mean_R, BDI_mean_R, Age_mean_R, Onset_mean_R, GAF_mean_R, OCI_mean_R]
    means_nr = [YBOCS_mean_NR, BDI_mean_NR, Age_mean_NR, Onset_mean_NR, GAF_mean_NR, OCI_mean_NR]
    
    ## turn correlations obtained above into covariances
    
    cov_YB_R = ttc(cor=cor_YB, sd=[YBOCS_sd_R, BDI_sd_R])
    cov_YA_R = ttc(cor=cor_YA, sd=[YBOCS_sd_R, Age_sd_R])
    cov_YO_R = ttc(cor=cor_YO, sd=[YBOCS_sd_R, Onset_sd_R])
    cov_YG_R = ttc(cor=cor_YG, sd=[YBOCS_sd_R, GAF_sd_R])
    cov_YOC_R = ttc(cor=cor_YOC, sd=[YBOCS_sd_R, OCI_sd_R])
    
    cov_BA_R = ttc(cor=cor_BA, sd=[BDI_sd_R, Age_sd_R])
    cov_BO_R = ttc(cor=cor_BO, sd=[BDI_sd_R, Onset_sd_R])
    cov_BG_R = ttc(cor=cor_BG, sd=[BDI_sd_R, GAF_sd_R])
    cov_BOC_R = ttc(cor=cor_BOC, sd=[BDI_sd_R, OCI_sd_R])
    
    cov_AO_R = ttc(cor=cor_AO, sd=[Age_sd_R, Onset_sd_R])
    cov_AG_R = ttc(cor=cor_AG, sd=[Age_sd_R, GAF_sd_R])
    cov_AOC_R = ttc(cor=cor_AOC, sd=[Age_sd_R, OCI_sd_R])
    
    cov_OG_R = ttc(cor=cor_OG, sd=[Onset_sd_R, GAF_sd_R])
    cov_OOC_R = ttc(cor=cor_OOC, sd=[Onset_sd_R, OCI_sd_R])
    
    cov_GOC_R = ttc(cor=cor_GOC, sd=[GAF_sd_R, OCI_sd_R])
    
    
    cov_YB_NR = ttc(cor=cor_YB, sd=[YBOCS_sd_NR, BDI_sd_NR])
    cov_YA_NR = ttc(cor=cor_YA, sd=[YBOCS_sd_NR, Age_sd_NR])
    cov_YO_NR = ttc(cor=cor_YO, sd=[YBOCS_sd_NR, Onset_sd_NR])
    cov_YG_NR = ttc(cor=cor_YG, sd=[YBOCS_sd_NR, GAF_sd_NR])
    cov_YOC_NR = ttc(cor=cor_YOC, sd=[YBOCS_sd_NR, OCI_sd_NR])
    
    cov_BA_NR = ttc(cor=cor_BA, sd=[BDI_sd_NR, Age_sd_NR])
    cov_BO_NR = ttc(cor=cor_BO, sd=[BDI_sd_NR, Onset_sd_NR])
    cov_BG_NR = ttc(cor=cor_BG, sd=[BDI_sd_NR, GAF_sd_NR])
    cov_BOC_NR = ttc(cor=cor_BOC, sd=[BDI_sd_NR, OCI_sd_NR])
    
    cov_AO_NR = ttc(cor=cor_AO, sd=[Age_sd_NR, Onset_sd_NR])
    cov_AG_NR = ttc(cor=cor_AG, sd=[Age_sd_NR, GAF_sd_NR])
    cov_AOC_NR = ttc(cor=cor_AOC, sd=[Age_sd_NR, OCI_sd_NR])
    
    cov_OG_NR = ttc(cor=cor_OG, sd=[Onset_sd_NR, GAF_sd_NR])
    cov_OOC_NR = ttc(cor=cor_OOC, sd=[Onset_sd_NR, OCI_sd_NR])
    
    cov_GOC_NR = ttc(cor=cor_GOC, sd=[GAF_sd_NR, OCI_sd_NR])

    ## turn obtained sd´s into variances
    
    YBOCS_var_R = YBOCS_sd_R**2
    BDI_var_R = BDI_sd_R**2
    Age_var_R = Age_sd_R**2
    Onset_var_R = Onset_sd_R**2
    GAF_var_R = GAF_sd_R**2
    OCI_var_R = OCI_sd_R**2
    
    YBOCS_var_NR = YBOCS_sd_NR**2
    BDI_var_NR = BDI_sd_NR**2
    Age_var_NR = Age_sd_NR**2
    Onset_var_NR = Onset_sd_NR**2
    GAF_var_NR = GAF_sd_NR**2
    OCI_var_NR = OCI_sd_NR**2
    
    ## create covariance matrices for responder/non responder
    
    covs_r = np.array([[YBOCS_var_R, cov_YB_R, cov_YA_R, cov_YO_R, cov_YG_R, cov_YOC_R], 
                       [cov_YB_R, BDI_var_R, cov_BA_R, cov_BO_R, cov_BG_R, cov_BOC_R], 
                       [cov_YA_R, cov_BA_R, Age_var_R, cov_AO_R, cov_AG_R, cov_AOC_R], 
                       [cov_YO_R, cov_BO_R, cov_AO_R, Onset_var_R, cov_OG_R, cov_OOC_R],
                       [cov_YG_R, cov_BG_R, cov_AG_R, cov_OG_R, GAF_var_R, cov_GOC_R],
                       [cov_YOC_R, cov_BOC_R, cov_AOC_R, cov_OOC_R, cov_GOC_R, OCI_var_R]
                       ])
    
    covs_nr = np.array([[YBOCS_var_NR, cov_YB_NR, cov_YA_NR, cov_YO_NR, cov_YG_NR, cov_YOC_NR], 
                       [cov_YB_NR, BDI_var_NR, cov_BA_NR, cov_BO_NR, cov_BG_NR, cov_BOC_NR], 
                       [cov_YA_NR, cov_BA_NR, Age_var_NR, cov_AO_NR, cov_AG_NR, cov_AOC_NR], 
                       [cov_YO_NR, cov_BO_NR, cov_AO_NR, Onset_var_NR, cov_OG_NR, cov_OOC_NR],
                       [cov_YG_NR, cov_BG_NR, cov_AG_NR, cov_OG_NR, GAF_var_NR, cov_GOC_NR],
                       [cov_YOC_NR, cov_BOC_NR, cov_AOC_NR, cov_OOC_NR, cov_GOC_NR, OCI_var_NR]
                       ])
    
    
    if RL == False: 
        
        ## since RL is False OL approach is used (simulation of only the variables obtained from the literature)
        
        Generator = np.random.default_rng(i_seed)
        
        samples_res = Generator.multivariate_normal(means_r, covs_r, size=size_res)
        samples_non_res = Generator.multivariate_normal(means_nr, covs_nr, size=size_non_res)

        # Create dataframes from the generated samples
        dat_res = pd.DataFrame(samples_res)
        dat_non_res = pd.DataFrame(samples_non_res)

        # create labels

        dat_res["res"] = rep(1,size_res)
        dat_non_res["res"] = rep(0,size_non_res)
        
        dat = pd.concat([dat_res, dat_non_res], axis=0)
        
        dat = dat.sample(frac=1.0, random_state=i)
        
        # extract labels
        
        res = dat['res'].values
        dat = dat.drop('res', axis=1)
        
        dat = np.round(dat)
        
        names = ["YBOO02T0", "BD2SUMT0","Age", "Beginn_Zwang_DSM", "GAF", "OCISUMT0"]
        dat.columns = names
        
        
    else: 
        
        ## since RL = True, RL is used -> the summary statistics of all variables in the data are added to the six variables obtained via the literature
        
        ## add label variable to X_train so the groups can be separated
        
        X_train["label"] = y_train
        
        resp = X_train[X_train["label"]==1]
        non_resp = X_train[X_train["label"]==0]
        
        ## remove label since this information is also not used in OL
        
        resp = resp.drop(columns=["label"])
        non_resp = non_resp.drop(columns=["label"])
        
        ## obtain mean and covariance structure
        
        resp_means = resp.mean()
        non_resp_means = non_resp.mean()
        
        resp_covs = resp.cov()
        non_resp_covs = non_resp.cov()
        
        ## replace obtained values with the simulated values where they are available
        
        resp_means[:6] = means_r
        non_resp_means[:6] = means_nr
        
        resp_covs.iloc[:6,:6] = covs_r
        non_resp_covs.iloc[:6,:6] = covs_nr
        
        ## create simulated data
        
        Generator = np.random.default_rng(i_seed)
        
        samples_res = Generator.multivariate_normal(resp_means, resp_covs, size=size_res)
        samples_non_res = Generator.multivariate_normal(non_resp_means, non_resp_covs, size=size_non_res)

        # Create dataframes from the generated samples
        dat_res = pd.DataFrame(samples_res)
        dat_non_res = pd.DataFrame(samples_non_res)

        # create labels

        dat_res["res"] = rep(1,size_res)
        dat_non_res["res"] = rep(0,size_non_res)
        
        dat = pd.concat([dat_res, dat_non_res], axis=0)
        
        dat = dat.sample(frac=1.0, random_state=i)
        
        # extract labels
        
        res = dat['res'].values
        dat = dat.drop('res', axis=1)
        
        ## extract column names and assign to dat
                
        names = X_train.drop(columns=["label"])
        dat.columns = names.columns
    
    return dat, res

def do_grid_search(X_train, y_train, RL_X, RL_y,i):
    "perform 3 grid searches, one for each available dataset (Real, OL, RL)"
    
    ## Hyperparameters grid search
    
    folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=i) # 5 because default -> realistic
    
    train_scoring = 'balanced_accuracy'
    
    rf_params = {"max_features": ["sqrt", "log2", 4,5],
                 "min_samples_leaf": [1,3,5,10], 
                 "max_samples": [0.66,0.8,None] 
                 }
    
    # preforming the actual grid search
    
    ## Real

    grid_search_rf_real = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestClassifier(random_state=i),
                           param_grid=rf_params,
                           cv=folds,
                           scoring=train_scoring,
                           verbose=0,
                           refit=False,
                           n_jobs=2)
    grid_search_rf_real.fit(X_train, y_train)
    best_rf_params_real = grid_search_rf_real.best_params_

    ## RL

    grid_search_rf_RL = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestClassifier(random_state=i),
                           param_grid=rf_params,
                           cv=folds,
                           scoring=train_scoring,
                           verbose=0,
                           refit=False,
                           n_jobs=2)
    grid_search_rf_RL.fit(RL_X,RL_y)
    best_rf_params_RL = grid_search_rf_RL.best_params_
    
    return best_rf_params_real, best_rf_params_RL

def get_mean_rate (fprs, tprs):
    "takes the mean values for each threshold"

    ## calculate mean
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    
    return mean_fpr, mean_tpr

def transform_proba_predict(tree_prob):
    "Transform the results of predict_proba() into a definite prediction, used in pretrain function"
    
    prob_arr = np.mean(tree_prob, axis=0)
    predictions = []
    for i in prob_arr:
        if i > 0.5:  
            predictions.append(1)   
        else:   
            predictions.append(0)
    return predictions, prob_arr

def impute(X_train, X_test, binarys, i):
    "impute numeric/on binary categoricals with regressor and binarys with classifier, both using MICE"
    
    # separate data into numeric/non binary categorical and binarys
    
    column_order = X_train.columns
    
    X_train_reg = X_train.loc[:, ~X_train.columns.isin(binarys["Variable_name"])]
    X_train_cat = X_train.loc[:, X_train.columns.isin(binarys["Variable_name"])]
    
    X_test_reg = X_test.loc[:, ~X_test.columns.isin(binarys["Variable_name"])]
    X_test_cat = X_test.loc[:, X_test.columns.isin(binarys["Variable_name"])]
    
    # fit estimator for numeric/non binary and impute
    
    imputer_reg = IterativeImputer(estimator = RandomForestRegressor(n_estimators=est_imput, random_state=i),random_state=i, max_iter=it_impute)
    
    train_reg_imputed = imputer_reg.fit_transform(X_train_reg)
    
    test_reg_imputed = imputer_reg.transform(X_test_reg) 

    X_reg_train = pd.DataFrame(train_reg_imputed, columns=X_train_reg.columns)
    X_reg_test_2 = pd.DataFrame(test_reg_imputed, columns=X_test_reg.columns)
    
    # round categoricals
    
    vector = ["BSIMITT0", "BSIM00T0", "BSIM01T0", "BSIM02T0", "BSIM03T0", "BSIM04T0", "BSIM05T0", "BSIM06T0", "BSIM07T0", "BSIM08T0", "BSIM09T0", "YBOSUMT0"]
    cols_to_round = [col for col in X_train_reg.columns if col not in vector]
    
    X_train_reg[cols_to_round] = X_train_reg[cols_to_round].round()

    X_reg_test_2[cols_to_round] = X_reg_test_2[cols_to_round].round()
    
    # fit imputer for binarys
    
    imputer_cat = IterativeImputer(estimator = RandomForestClassifier(n_estimators=est_imput, random_state=i),random_state=i, max_iter=it_impute)
    
    train_cat_imputed = imputer_cat.fit_transform(X_train_cat)
    
    test_cat_imputed = imputer_cat.transform(X_test_cat)

    X_cat_train = pd.DataFrame(train_cat_imputed, columns=X_train_cat.columns)
    X_cat_test_2 = pd.DataFrame(test_cat_imputed, columns=X_test_cat.columns)
    
    X_train = pd.concat([X_cat_train, X_reg_train], axis=1)
    X_train = X_train.reindex(columns=column_order)
    
    X_test_2 = pd.concat([X_cat_test_2, X_reg_test_2], axis=1)
    X_test_2 = X_test_2.reindex(columns=column_order)
    
    return X_train, X_test_2

def pretrain (X_test,y_test,X_train, y_train, s_X, s_y, n_pre, n_real, i, params_pre, params_fine, RL=False): 
    "used for OL pretraining, creates two random forests and combines their votes into a final prediction"
    
    ## train random forest with OL simulated data
    
    rf = RandomForestClassifier(n_estimators=n_pre, random_state=i)
    rf.set_params(**params_pre)
    rf.fit(s_X, s_y)
    
    ## train random forest with real data
    
    rf_2 = RandomForestClassifier(n_estimators=n_real, random_state=i + 5)
    rf_2.set_params(**params_fine)
    rf_2.fit(X_train, y_train)
    
    ## retrieve probability from "pretrained" trees for every prediction, depending on wether RL is used, only simulated variables from X_test are used
    
    pre_tree_prob = []
    
    if RL == False: 
        for tree in rf.estimators_ :
            pre_tree_prob.append(tree.predict_proba(X_test[simulated])[:, 1])
    else: 
        for tree in rf.estimators_ :
            pre_tree_prob.append(tree.predict_proba(X_test)[:, 1])
    
        
    ## retrieve probability from "fine-tuned" trees for every prediction
    
    real_tree_prob = []
    for tree in rf_2.estimators_ :
            real_tree_prob.append(tree.predict_proba(X_test)[:, 1])
            

    pre_tree_prob.extend(real_tree_prob)
    
    final_pred, prob_arr = transform_proba_predict(pre_tree_prob)
    

    tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()

    ## calculate accuracys
    
    accuracy = balanced_accuracy_score(y_test, final_pred)
    
    return accuracy,prob_arr, tn, fp, fn, tp

X= load_data()

X = X.dropna(subset=["YBOCS_post"])

pre_col_num = X.columns.get_loc('YBOO02T0')
post_col_num = X.columns.get_loc('YBOCS_post')

X.reset_index(drop=True, inplace=True)

X = binarise(X, pre_col_num, post_col_num)

labels = X["resp"]
y = pd.DataFrame(labels)

# here MADRS is also included since otherwise the imputation is slightly differnet due to ordering effects

simulated_names = ["YBOO02T0", "BD2SUMT0", "Age", "Beginn_Zwang_DSM", "GAF", "OCISUMT0", "MADSUMT0"]

X = X[simulated_names + [col for col in X.columns if col not in simulated_names]]

X.drop(columns=['YBOCS_post', 'resp', "RCI"], inplace=True)

simulated = ["YBOO02T0", "BD2SUMT0","Age", "Beginn_Zwang_DSM", "GAF", "OCISUMT0"]

## create lists to store obtained performance measures

accuracys = []
accuracys_SMOTE = []
accuracys_train = []
accuracys_30 = []
accuracys_70 = []
probs = []
preds = []

accuracys_RL_100 = []
RL_100_fns = []

accuracys_RL_100_sm = []
RL_100_fns_sm = []

n_y = []

tn_stan =[]
fp_stan = []
fn_stan = []
tp_stan = []

tn_stan_SMOTE =[]
fp_stan_SMOTE = []
fn_stan_SMOTE = []
tp_stan_SMOTE = []

tn_pre =[]
fp_pre = []
fn_pre = []
tp_pre = []

tn_pre_SMOTE =[]
fp_pre_SMOTE = []
fn_pre_SMOTE = []
tp_pre_SMOTE = []

probs_RL_100 = []
preds_RL_100 = []

# create thresholds to evaluate calssifiers on, to get same number of thresholds so they can be plotted together

threshold = np.arange(1, 0, -0.01)

## import excels containing values from the literature for simulation

binary_import_path = ""

binarys = pd.read_excel(binary_import_path)

findings_import_path = ""
cor_import_path = ""

findings = pd.read_excel(findings_import_path)
cor_file = pd.read_excel(cor_import_path)


for i in range(0,iterations):
    
    print("Iteration " + str(i+1))

    ## conduct train test split    

    X_train, X_test, y_train, y_test = do_traintestsplit(X, y, i) 
    
    # set simulated sample sizes 
    
    size_res = 500
    size_non_res = 500
    
    ## imputation using CART
    
    X_train, X_test_2 = impute(X_train, X_test, binarys, i)
    
    RL_X, RL_y = SIMCOR (findings,cor_file, number_of_Variables_1 = range(0,15),number_of_Variables_2 = range(0,6), RL=True, X_train = X_train, y_train = y_train, i_seed = i)

    X_train = X_train.drop(X_train.columns[501], axis=1)    

    ## conduct grid search (for the real data this is done before SMOTEing because the hyperparameters shall help the classifier characterizing the real datas distribution not a SMOTEed version of it)
    
    best_rf_params_real, best_rf_params_RL = do_grid_search(X_train, y_train, RL_X, RL_y, i)
    
    ## fit standard random forest to the data and save performance measures

    
    accuracy, accuracy_train, y_pred_prob, accuracy_30, accuracy_70, tn, fp, fn, tp = do_predictions(X_train = X_train, X_test = X_test_2, y_train = y_train, y_test = y_test, i = i, params = best_rf_params_real)
    
    accuracys.append(accuracy)
    accuracys_train.append(accuracy_train)
    accuracys_30.append(accuracy_30)
    accuracys_70.append(accuracy_70)
    probs.append(y_pred_prob)
    preds.append(y_test)
    
    tn_stan.append(tn/len(y_test))
    fp_stan.append(fp/len(y_test))
    fn_stan.append(fn/len(y_test))
    tp_stan.append(tp/len(y_test))
    

    accuracy_rl, y_pred_prob, tn, fp, fn, tp = pretrain(X_test_2, y_test,X_train, y_train, RL_X, RL_y, 100, 100, i, best_rf_params_RL, best_rf_params_real, RL=True)
            
    accuracys_RL_100.append(accuracy_rl) 
    probs_RL_100.append(y_pred_prob)
    preds_RL_100.append(y_test)
    
    tn_pre.append(tn/len(y_test))
    fp_pre.append(fp/len(y_test))
    fn_pre.append(fn/len(y_test))
    tp_pre.append(tp/len(y_test))

    ## apply SMOTE

    vector = ["BSIMITT0", "BSIM00T0", "BSIM01T0", "BSIM02T0", "BSIM03T0", "BSIM04T0", "BSIM05T0", "BSIM06T0", "BSIM07T0", "BSIM08T0", "BSIM09T0", "YBOSUMT0"]
    cat_columns_smote = [col for col in X_train.columns if col not in vector]
    sm = SMOTENC(random_state=i, categorical_features=cat_columns_smote)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    print('Resampled dataset shape %s' % Counter(y_train))
    
    accuracy, accuracy_train, y_prob_mean, accuracy_30, accuracy_70, tn, fp, fn, tp = do_predictions(X_train = X_train, X_test = X_test_2, y_train = y_train, y_test = y_test, i = i, params = best_rf_params_real)
    
    accuracys_SMOTE.append(accuracy)
    
    tn_stan_SMOTE.append(tn/len(y_test))
    fp_stan_SMOTE.append(fp/len(y_test))
    fn_stan_SMOTE.append(fn/len(y_test))
    tp_stan_SMOTE.append(tp/len(y_test))
    
    accuracy_rl_sm,y_pred_2, tn, fp, fn, tp = pretrain(X_test_2, y_test,X_train, y_train, RL_X, RL_y, 100, 100, i, best_rf_params_RL, best_rf_params_real, RL=True)
            
    accuracys_RL_100_sm.append(accuracy_rl_sm) 
    
    tn_pre_SMOTE.append(tn/len(y_test))
    fp_pre_SMOTE.append(fp/len(y_test))
    fn_pre_SMOTE.append(fn/len(y_test))
    tp_pre_SMOTE.append(tp/len(y_test))
    
    n_y.append(len(y_test))
    


## Create calibration plot for standard


num_bins = 10
merged_probs = []
merged_preds = []

for i in range(0,len(probs)):
    
    merged = pd.DataFrame({"probs": probs[i], "preds": preds[i]})
    merged = merged.sort_values(by='probs', ascending=True)
    merged_probs.append(merged["probs"])
    merged_preds.append(merged["preds"])


max_length = max(probs.shape[0] for probs in merged_probs)

# Add NaNs to make all arrays the same lengt
merged_probs = [np.pad(probs, (0, max_length - probs.shape[0]), mode='constant', constant_values=np.nan) for probs in merged_probs]

merged_preds = [np.pad(preds, (0, max_length - preds.shape[0]), mode='constant', constant_values=np.nan) for preds in merged_preds]

# take mean ignoring na´s
mean_probs = np.nanmean(merged_probs, axis = 0)
mean_preds = np.nanmean(merged_preds, axis = 0)

bin_boundaries = np.linspace(mean_probs.min(), mean_probs.max(), num_bins + 1)

# Map the values to the bins
bin_indices = np.digitize(mean_probs, bin_boundaries, right=True)

merged_pred_probs = pd.DataFrame({"mean preds": mean_preds, "mean probs": mean_probs, "bin": bin_indices})

avg_preds = []

for i in np.unique(bin_indices): 
    
    selected_rows = merged_pred_probs.loc[merged_pred_probs['bin'] == i]
    avg_preds.append(np.mean(selected_rows["mean preds"]))

plt.plot(np.unique(bin_indices), avg_preds, marker='o', label='Calibration function', color = "black")
plt.plot([0, 10], [0, 1], linestyle='--', label='Perfectly Calibrated', color = "grey")
plt.xticks(ticks = [0,2,4,6,8,10], labels=["0", "0.2", "0.4", "0.6", "0.8", "1"])

# Add labels and title
plt.xlabel('Predicted probability')
plt.ylabel('Fraction of Positives')
plt.title('Average reliability diagram over 100 Iterations')
plt.legend()
plt.savefig('', dpi=300, bbox_inches='tight')
plt.close()

## plot histogram

plt.xlabel('Mean predicted probability')
plt.ylabel('Count')
plt.hist(mean_probs, 
         range = (0,1), 
         bins = 10,
         color = "dimgrey")

plt.show()
plt.savefig('', dpi=300, bbox_inches='tight')
plt.close()

## plot confusion matrices

# standard

confusion_matrix = np.array([[np.mean(tn_stan), np.mean(fp_stan)], [np.mean(fn_stan), np.mean(tp_stan)]])

sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot(colorbar=0, cmap = "Greys")
plt.savefig('', dpi=300, bbox_inches='tight')

# pretraining

confusion_matrix = np.array([[np.mean(tn_pre), np.mean(fp_pre)], [np.mean(fn_pre), np.mean(tp_pre)]])

sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot(colorbar=0, cmap = "Greys")
plt.savefig('', dpi=300, bbox_inches='tight')

# standard SMOTE

confusion_matrix = np.array([[np.mean(tn_stan_SMOTE), np.mean(fp_stan_SMOTE)], [np.mean(fn_stan_SMOTE), np.mean(tp_stan_SMOTE)]])

sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot(colorbar=0, cmap = "Greys")
plt.savefig('', dpi=300, bbox_inches='tight')

# pretraining SMOTE


confusion_matrix = np.array([[np.mean(tn_pre_SMOTE), np.mean(fp_pre_SMOTE)], [np.mean(fn_pre_SMOTE), np.mean(tp_pre_SMOTE)]])

sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot(colorbar=0, cmap = "Greys")
plt.savefig('', dpi=300, bbox_inches='tight')

plt.close()



results = pd.DataFrame({
    "Balanced Accuracy": [np.mean(accuracys),np.mean(accuracys_SMOTE), np.mean(accuracys_RL_100), np.mean(accuracys_RL_100_sm)],
    "Acc on train": [np.mean(accuracys_train), "None", "None", "None"],
    "mean data test": [np.mean(n_y), np.mean(n_y), np.mean(n_y), np.mean(n_y)],
    "mean tn": [np.mean(tn_stan), np.mean(tn_stan_SMOTE), np.mean(tn_pre), np.mean(tn_pre_SMOTE) ],
    "mean fp": [np.mean(fp_stan), np.mean(fp_stan_SMOTE), np.mean(fp_pre), np.mean(fp_pre_SMOTE) ],
    "mean fn": [np.mean(fn_stan), np.mean(fn_stan_SMOTE), np.mean(fn_pre), np.mean(fn_pre_SMOTE)],
    "mean tp": [np.mean(tp_stan), np.mean(tp_stan_SMOTE), np.mean(tp_pre), np.mean(tp_pre_SMOTE) ]
}, index=["standard without smote", "standard with SMOTE", "unSMOTEed pretrained", "SMOTEed pretrained"])

results.to_excel("")

print("mean accuracy: " + str(np.mean(accuracys)))
print("mean accuracy train_ " + str(np.mean(accuracys_train)))
print("mean data in test: " + str(np.mean(n_y)))
print("mean fn for unSMOTEed pretrained: " + str(np.mean(RL_100_fns)))
print("mean fn for SMOTEed pretrained: " + str(np.mean(RL_100_fns_sm)))








