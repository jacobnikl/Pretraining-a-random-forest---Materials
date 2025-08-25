#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:06:44 2023

@author: niklasjacobs
"""

import pandas as pd
import numpy as np
import copy
import sklearn as sklearn
import warnings
import matplotlib.pyplot as plt
import math
import miceforest as mf
import datetime as dt
import scipy
import random
from multiprocessing import Pool
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from scipy.stats import t
from imblearn.over_sampling import SMOTENC
from scipy.stats import multivariate_normal
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings("ignore")

execution_start = dt.datetime.now()

pd.set_option('display.max_rows', None)

threshold = np.arange(1, 0, -0.01)

# set simulated sample sizes 
size_res = 500
size_non_res = 500
    
simulated = ["YBOO02T0", "BD2SUMT0", "Age", "Beginn_Zwang_DSM", "GAF"]  

number_iterations = 100

def load_data():
    """Import patient Data and string containing used variables"""

    features_import_path = "data/real/combined_data_remission.xlsx"
    features_import = pd.read_excel(features_import_path)
    
    string_import_path = "data/real/used_string_check_stan.csv"
    string_import = read_csv(string_import_path, sep=";", header=0)
    
    string_import = string_import.drop(string_import.columns[0], axis=1)
    string_import = string_import.drop(string_import.columns[502:1000], axis=1)

    common_columns = string_import.columns
    features_import = features_import[common_columns]
     
    labels = features_import["remission"]
    labels_import = pd.DataFrame(labels)

    features_import.drop(columns=['remission'], inplace=True)
    
    # dropping some remaining NA´s
    
    features_import.iloc[:, [497, 498, 499, 500]] = features_import.iloc[:, [497, 498, 499, 500]].fillna(777)
    
    
    ## reordering of datasets in order to be able to obtain the variables of interest in pretrain based on index later on
    
    new_order = ["YBOO02T0", "BD2SUMT0", "Age", "Beginn_Zwang_DSM", "GAF"]

    # Add remaining columns that are not in new_order
    remaining_columns = [col for col in features_import.columns if col not in new_order]

    # Combine both lists
    final_order = new_order + remaining_columns

    # Reorder DataFrame
    features_import = features_import[final_order]
    

    return features_import, labels_import

def rep(value, n):
    "function to paste a value (for instance mean) n times, used for weighting coefficients in simcor"
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

def do_traintestsplit(X,y,i):
    """Conduct a single train-test split with 20% of data in test """
    
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, test_size=0.3, random_state=i)
    
    return X_train, X_test, y_train, y_test

def threshold_predictions(y_prob, threshold):
    return (y_prob >= threshold).astype(int)
            
def do_predictions(X_train, X_test, y_train, y_test, i, params):
    """Train standard Random Forest classifier on train split and get predictions on test data"""
            
    clf = RandomForestClassifier(n_estimators=200, random_state=i)
    clf.set_params(**params)
    clf = clf.fit(X_train, y_train)
    accuracy_s = balanced_accuracy_score(y_test, clf.predict(X_test))
    
    sensitivity, specificity = get_sens_spec(y_true=y_test, y_pred=clf.predict(X_test))
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]


    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    importances = clf.feature_importances_
    feature_names = list(X_train.columns)
    importances_df = pd.DataFrame({'feature_names': feature_names, 'importances': importances})

    # Sort the dataframe by feature importance value
    importances_df = importances_df.sort_values('importances', ascending=False)

    
    return accuracy_s, sensitivity, specificity, roc_auc, threshold, importances_df

def corrected_t_test(accuracys_sim, accuracys, sample_training, sample_test): 
    "corrected resampled t test according to Bouackert and Frank"
    
    diff = [accuracys_sim - accuracys for accuracys_sim, accuracys in zip(accuracys_sim, accuracys)]
    d_mean = np.mean(diff)
    sigma = np.var(diff)
    n1 = sample_training
    n2 = sample_test
    n = len(diff)
    sigma2_mod = sigma * (1/n + n2/n1)
    t_statistic =  d_mean / np.sqrt(sigma2_mod)
    Pvalue = t.sf(abs(t_statistic), n-1)
    
    return t_statistic, Pvalue
        
        

def get_sens_spec (y_true, y_pred):
    "extract sensitivity and specificity from test labels and predicted labels"
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    return sensitivity, specificity

def replace_nan_with_weighted_mean(df, value_col, weight_col, condition_col, condition_value):
    # Filter the DataFrame for rows where the condition is met
    filtered_df = df[df[condition_col] == condition_value]
    filtered_df = filtered_df.dropna()
    
    # Calculate the weighted mean
    weighted_mean = np.average(filtered_df[value_col], weights=filtered_df[weight_col])
    
    # Replace NaN values in the specified column with the weighted mean
    
    df.loc[df[condition_col] == condition_value, value_col] = df.loc[df[condition_col] == condition_value, value_col].fillna(weighted_mean)
    #df[value_col].fillna(weighted_mean, inplace=True)
    
    return df


def SIMCOR (findings, cor_file,number_of_Variables_1, number_of_Variables_2, RL, X_train, y_train, i_seed):
    "extract values found in the literature from excels, then simulate data using these values"

    ## define lists to store the correlations
    
    cor_YB = []
    cor_YA = []
    cor_YO = []
    cor_YG = []
    cor_BA = []
    cor_BO = []
    cor_BG = []
    cor_AO = []
    cor_AG = []
    cor_OG = []


    
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
                                                if current.iloc[0,0] == "OG":
                                                        
                                                    OG_cor_create = []
                                                        
                                                    OG_cor_create.append(add_cor)
                                                        
                                                    for sublist in OG_cor_create:
                                                        cor_OG.extend(sublist)
                                                                
                                                                
                                
    
    ## get the mean value of the list -> correlations weighted by n                                                

    cor_YB = sum(cor_YB)/len(cor_YB)
    cor_YA = sum(cor_YA)/len(cor_YA)
    cor_YO = sum(cor_YO)/len(cor_YO)
    cor_YG = sum(cor_YG)/len(cor_YG)
    cor_BA = sum(cor_BA)/len(cor_BA)
    cor_BO = sum(cor_BO)/len(cor_BO)
    cor_BG = sum(cor_BG)/len(cor_BG)
    cor_AO = sum(cor_AO)/len(cor_AO)
    cor_AG = sum(cor_AG)/len(cor_AG)
    cor_OG = sum(cor_OG)/len(cor_OG)

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
    
    ## define a variable that contains every variable name only once
    
    unique_entries = findings['Variable '].unique()

    ## extract variable that matches the unique entry accessed by i    

    for i in number_of_Variables_2:
        
        VARIABLE = findings[findings['Variable '] == unique_entries[i]]
        
        for i in range(len(VARIABLE)):
            
            ## extract means and standard deviations for the currently accessed row
            
            current = VARIABLE.iloc[[i]]
            current = current.T
            
            ## use rep to past the value as often as its n
            
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
                                
                                        

    ## get mean values od the lists -> means and sd weighted by n                  
     
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
    
  
    
    ## create vector with the obtained means
    
    means_r = [YBOCS_mean_R, BDI_mean_R, Age_mean_R, Onset_mean_R, GAF_mean_R]
    means_nr = [YBOCS_mean_NR, BDI_mean_NR, Age_mean_NR, Onset_mean_NR, GAF_mean_NR]
    
    ## turn correlations obtained above into covariances
    
    cov_YB_R = ttc(cor=cor_YB, sd=[YBOCS_sd_R, BDI_sd_R])
    cov_YA_R = ttc(cor=cor_YA, sd=[YBOCS_sd_R, Age_sd_R])
    cov_YO_R = ttc(cor=cor_YO, sd=[YBOCS_sd_R, Onset_sd_R])
    cov_YG_R = ttc(cor=cor_YG, sd=[YBOCS_sd_R, GAF_sd_R])

    
    cov_BA_R = ttc(cor=cor_BA, sd=[BDI_sd_R, Age_sd_R])
    cov_BO_R = ttc(cor=cor_BO, sd=[BDI_sd_R, Onset_sd_R])
    cov_BG_R = ttc(cor=cor_BG, sd=[BDI_sd_R, GAF_sd_R])

    
    cov_AO_R = ttc(cor=cor_AO, sd=[Age_sd_R, Onset_sd_R])
    cov_AG_R = ttc(cor=cor_AG, sd=[Age_sd_R, GAF_sd_R])

    
    cov_OG_R = ttc(cor=cor_OG, sd=[Onset_sd_R, GAF_sd_R])


    
    
    cov_YB_NR = ttc(cor=cor_YB, sd=[YBOCS_sd_NR, BDI_sd_NR])
    cov_YA_NR = ttc(cor=cor_YA, sd=[YBOCS_sd_NR, Age_sd_NR])
    cov_YO_NR = ttc(cor=cor_YO, sd=[YBOCS_sd_NR, Onset_sd_NR])
    cov_YG_NR = ttc(cor=cor_YG, sd=[YBOCS_sd_NR, GAF_sd_NR])

    
    cov_BA_NR = ttc(cor=cor_BA, sd=[BDI_sd_NR, Age_sd_NR])
    cov_BO_NR = ttc(cor=cor_BO, sd=[BDI_sd_NR, Onset_sd_NR])
    cov_BG_NR = ttc(cor=cor_BG, sd=[BDI_sd_NR, GAF_sd_NR])

    
    cov_AO_NR = ttc(cor=cor_AO, sd=[Age_sd_NR, Onset_sd_NR])
    cov_AG_NR = ttc(cor=cor_AG, sd=[Age_sd_NR, GAF_sd_NR])

    
    cov_OG_NR = ttc(cor=cor_OG, sd=[Onset_sd_NR, GAF_sd_NR])

    


    
    ## turn obtained sd´s into variances
    
    YBOCS_var_R = YBOCS_sd_R**2
    BDI_var_R = BDI_sd_R**2
    Age_var_R = Age_sd_R**2
    Onset_var_R = Onset_sd_R**2
    GAF_var_R = GAF_sd_R**2
    
    YBOCS_var_NR = YBOCS_sd_NR**2
    BDI_var_NR = BDI_sd_NR**2
    Age_var_NR = Age_sd_NR**2
    Onset_var_NR = Onset_sd_NR**2
    GAF_var_NR = GAF_sd_NR**2
    
    ## create covariance matrices for responder/non responder
    
    covs_r = np.array([[YBOCS_var_R, cov_YB_R, cov_YA_R, cov_YO_R, cov_YG_R], 
                       [cov_YB_R, BDI_var_R, cov_BA_R, cov_BO_R, cov_BG_R], 
                       [cov_YA_R, cov_BA_R, Age_var_R, cov_AO_R, cov_AG_R], 
                       [cov_YO_R, cov_BO_R, cov_AO_R, Onset_var_R, cov_OG_R],
                       [cov_YG_R, cov_BG_R, cov_AG_R, cov_OG_R, GAF_var_R],
                       ])
    
    covs_nr = np.array([[YBOCS_var_NR, cov_YB_NR, cov_YA_NR, cov_YO_NR, cov_YG_NR], 
                       [cov_YB_NR, BDI_var_NR, cov_BA_NR, cov_BO_NR, cov_BG_NR], 
                       [cov_YA_NR, cov_BA_NR, Age_var_NR, cov_AO_NR, cov_AG_NR], 
                       [cov_YO_NR, cov_BO_NR, cov_AO_NR, Onset_var_NR, cov_OG_NR],
                       [cov_YG_NR, cov_BG_NR, cov_AG_NR, cov_OG_NR, GAF_var_NR],
                       ])
    
    
    if RL == False: 
        
        ## since RL is False OL approach is used (simulation of only the variables obtained in the literature)
        
        Generator = np.random.default_rng(i_seed)
        
        samples_res = Generator.multivariate_normal(means_r, covs_r, size=size_res)
        samples_non_res = Generator.multivariate_normal(means_nr, covs_nr, size=size_non_res)

        # Create dataframes from the generated samples
        dat_res = pd.DataFrame(samples_res)
        dat_non_res = pd.DataFrame(samples_non_res)

        # create labels

        dat_res["res"] = rep(2,size_res)
        dat_non_res["res"] = rep(1,size_non_res)
        
        dat = pd.concat([dat_res, dat_non_res], axis=0)
        
        dat = dat.sample(frac=1.0, random_state=i)
        
        # extract labels
        
        res = dat['res'].values
        dat = dat.drop('res', axis=1)
        
        #### replace names
        
        names = ["YBOO02T0", "BD2SUMT0","Age", "Beginn_Zwang_DSM", "GAF"]
        dat.columns = names
        
        
    else: 
        
        ## since RL = True, RL is used -> the summary statistics of all variables in the data are added to the seven variables obtained via the literature
        
        ## add label variable to X_train so the groups can be separated
        
        X_train["label"] = y_train
        
        resp = X_train[X_train["label"]==2]
        non_resp = X_train[X_train["label"]==1]
        
        ## remove label since this information is also not used in OL
        
        resp = resp.drop(columns=["label"])
        non_resp = non_resp.drop(columns=["label"])
        
        ## obtain mean and covariance structure
        
        resp_means = resp.mean()
        non_resp_means = non_resp.mean()
        
        resp_covs = resp.cov()
        non_resp_covs = non_resp.cov()

        
        ## replace obtained values with the simulated values where they are available
        
        resp_means[:5] = means_r
        non_resp_means[:5] = means_nr
        
        resp_covs.iloc[:5,:5] = covs_r
        non_resp_covs.iloc[:5,:5] = covs_nr
        
        ## create response variable and data frame to store simulated data
        
        Generator = np.random.default_rng(i_seed)
        
        samples_res = Generator.multivariate_normal(resp_means, resp_covs, size=size_res)
        samples_non_res = Generator.multivariate_normal(non_resp_means, non_resp_covs, size=size_non_res)

        # Create dataframes from the generated samples
        dat_res = pd.DataFrame(samples_res)
        dat_non_res = pd.DataFrame(samples_non_res)

        # create labels

        dat_res["res"] = rep(2,size_res)
        dat_non_res["res"] = rep(1,size_non_res)
        
        dat = pd.concat([dat_res, dat_non_res], axis=0)
        
        dat = dat.sample(frac=1.0, random_state=i)
        
        # extract labels
        
        res = dat['res'].values
        dat = dat.drop('res', axis=1)
        
        ## extract column names and assign to dat
                
        names = X_train.drop(columns=["label"])
        dat.columns = names.columns
    
    return dat, res

def do_grid_search(X_train, y_train, OL_X, OL_y, RL_X, RL_y):
    "perform 3 grid searches, one for each available dataset (Real, OL, RL)"
    
    ## Hyperparameters grid search
    
    folds = 5 # because default -> realistic
    
    train_scoring = 'balanced_accuracy'
    
    rf_params = {"max_features": ["sqrt", "log2", 4,5],
                 "min_samples_leaf": [1,3,5,10], 
                 "max_samples": [0.66,0.8,None] 
                 }
    
    ## Real

    grid_search_rf_real = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestClassifier(),
                           param_grid=rf_params,
                           cv=folds,
                           scoring=train_scoring,
                           verbose=0,
                           refit=False)
    grid_search_rf_real.fit(X_train, y_train)
    best_rf_params_real = grid_search_rf_real.best_params_

    ## OL 
    
    grid_search_rf_OL = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestClassifier(),
                           param_grid=rf_params,
                           cv=folds,
                           scoring=train_scoring,
                           verbose=0,
                           refit=False)
    grid_search_rf_OL.fit(OL_X,OL_y)
    best_rf_params_OL = grid_search_rf_OL.best_params_

    ## RL

    grid_search_rf_RL = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestClassifier(),
                           param_grid=rf_params,
                           cv=folds,
                           scoring=train_scoring,
                           verbose=0,
                           refit=False)
    grid_search_rf_RL.fit(RL_X,RL_y)
    best_rf_params_RL = grid_search_rf_RL.best_params_
    
    return best_rf_params_real, best_rf_params_OL, best_rf_params_RL

def get_mean_rate (fprs, tprs):
    "takes the mean values for each threshold"

    ## calculate mean
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    
    return mean_fpr, mean_tpr

def transform_proba_predict(tree_prob):
    "Transform the results of predict_proba() into a definite prediction"
    
    prob_arr = np.mean(tree_prob, axis=0)
    predictions = []
    for i in prob_arr:
        if i > 0.5:  
            predictions.append(2)   
        else:   
            predictions.append(1)
    return predictions

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
    
    ## retrieve probability from "pretrained" trees for every prediction
    
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
    
    final_pred = transform_proba_predict(pre_tree_prob)
    
    # Calculate the TPR and FPR for a specific threshold

    tprs = []
    fprs = []
    
    mean_pre_prob = np.mean(pre_tree_prob, axis=0)

    roc_auc = roc_auc_score(y_test, mean_pre_prob)
    
    ## calculate accuracys
    
    accuracy = balanced_accuracy_score(y_test, final_pred)
    sensitivity, specificity = get_sens_spec(y_test, final_pred)
    
    return accuracy, sensitivity, specificity, roc_auc

def impute(X_train, X_test, binarys, i):
    "impute numeric/on binary categoricals with regressor and binarys with classifier, both using MICE"
    
    # separate data into numeric/non binary categorical and binarys
    
    column_order = X_train.columns
    
    X_train_reg = X_train.loc[:, ~X_train.columns.isin(binarys)]
    X_train_cat = X_train.loc[:, X_train.columns.isin(binarys)]
    
    X_test_reg = X_test.loc[:, ~X_test.columns.isin(binarys)]
    X_test_cat = X_test.loc[:, X_test.columns.isin(binarys)]
    
    # fit estimator for numeric/non binary and impute
    
    imputer_reg = IterativeImputer(estimator = RandomForestRegressor(n_estimators=100, random_state=i),random_state=i)
    
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
    
    imputer_cat = IterativeImputer(estimator = RandomForestClassifier(n_estimators=100, random_state=i),random_state=i)
    
    train_cat_imputed = imputer_cat.fit_transform(X_train_cat)
    
    test_cat_imputed = imputer_cat.transform(X_test_cat)

    X_cat_train = pd.DataFrame(train_cat_imputed, columns=X_train_cat.columns)
    X_cat_test_2 = pd.DataFrame(test_cat_imputed, columns=X_test_cat.columns)
    
    X_train = pd.concat([X_cat_train, X_reg_train], axis=1)
    X_train = X_train.reindex(columns=column_order)
    
    X_test_2 = pd.concat([X_cat_test_2, X_reg_test_2], axis=1)
    X_test_2 = X_test_2.reindex(columns=column_order)
    
    return X_train, X_test_2



def perform_one_iteration(i):
    
    print("Iteration " + str(i+1))
    X, y = load_data()
    X.reset_index(drop=True, inplace=True)
    
    findings_import_path = "data/to_sim/Remmission_MW_sys.xlsx"
    cor_import_path = "data/to_sim/cor_data_rem.xlsx"
    findings = pd.read_excel(findings_import_path)
    cor_file = pd.read_excel(cor_import_path)

    # replace NaN with weighted mean
    findings = replace_nan_with_weighted_mean(findings, value_col='Mean_NR', weight_col='N', condition_col='Variable ', condition_value='YBOCS')
    findings = replace_nan_with_weighted_mean(findings, value_col='SD_NR', weight_col='N', condition_col='Variable ', condition_value='YBOCS')
    findings = replace_nan_with_weighted_mean(findings, value_col='Mean_NR', weight_col='N', condition_col='Variable ', condition_value='BDI')
    findings = replace_nan_with_weighted_mean(findings, value_col='SD_NR', weight_col='N', condition_col='Variable ', condition_value='BDI')

    ## conduct train test split    
    X_train, X_test, y_train, y_test = do_traintestsplit(X, y, i) 

    X_test_len_sin = len(X_test)
    X_train_len_sin = len(X_train)
    
    cols_only_one_value = [col for col in X_train.columns if X_train[col].nunique() == 1]
    X_train = X_train.drop(columns=cols_only_one_value)
    X_test = X_test.drop(columns=cols_only_one_value) 
    
    # drop categorical columns with less than 10% minority class
    binary_cols = [col for col in X_train.columns 
               if X_train[col].nunique(dropna=True) == 2]
    
    cols_to_drop = []
    for col in binary_cols:
        value_counts = X_train[col].value_counts(normalize=True, dropna=True)
        if len(value_counts) == 2:  
            minority_class_prop = value_counts.min()
            if minority_class_prop < 0.1:
                cols_to_drop.append(col)
        else: 
            cols_to_drop.append(col)


    # Drop the columns
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    
    ## MICE imputation using CART
    X_train, X_test_2 = impute(X_train, X_test, binary_cols, i)
    
    vector = ["BSIMITT0", "BSIM00T0", "BSIM01T0", "BSIM02T0", "BSIM03T0", "BSIM04T0", "BSIM05T0", "BSIM06T0", "BSIM07T0", "BSIM08T0", "BSIM09T0", "YBOSUMT0"]
    cols_to_round = [col for col in X_train.columns if col not in vector]
    
    X_train[cols_to_round] = X_train[cols_to_round].round()
    X_test[cols_to_round] = X_test[cols_to_round].round()
    
    X_train = pd.DataFrame(X_train, columns=X_train.columns)
    X_train_2 = X_train.copy()
    X_test_2 = pd.DataFrame(X_test, columns=X_test.columns)
    
    ## simulate data
    OL_X, OL_y = SIMCOR(findings, cor_file, number_of_Variables_1=range(0,10), number_of_Variables_2=range(0,5), RL=False, X_train=X_train_2, y_train=y_train, i_seed=i)
    RL_X, RL_y = SIMCOR(findings, cor_file, number_of_Variables_1=range(0,10), number_of_Variables_2=range(0,5), RL=True,  X_train=X_train_2, y_train=y_train, i_seed=i)
    
    # apply SMOTE
    vector = ["BSIMITT0", "BSIM00T0", "BSIM01T0", "BSIM02T0", "BSIM03T0", "BSIM04T0", "BSIM05T0", "BSIM06T0", "BSIM07T0", "BSIM08T0", "BSIM09T0", "YBOSUMT0"]
    cat_columns_smote = [col for col in X_train.columns if col not in vector]
    sm = SMOTENC(random_state=i, categorical_features=cat_columns_smote)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    ## grid search
    best_rf_params_real, best_rf_params_OL, best_rf_params_RL = do_grid_search(X_train, y_train, OL_X, OL_y, RL_X, RL_y)
    
    ## fit baseline RF
    accuracy, sensitivity, specificity, roc_auc, threshold, importances_df = do_predictions(
        X_train=X_train, X_test=X_test_2, y_train=y_train, y_test=y_test, i=i, params=best_rf_params_real
    )
    
    # pretraining
    i_first = i
    n_pre = [33,67,100]

    for n in n_pre:
        if n == 33:
            accuracy_ol_20, sensitivity_ol_20, specificity_ol_20, roc_auc_ol_20 = pretrain(
                X_test_2, y_test, X_train, y_train, OL_X[simulated], OL_y, n, 167,
                i_first, best_rf_params_OL, best_rf_params_real, RL=False
            )
            accuracy_rl_20, sensitivity_rl_20, specificity_rl_20, roc_auc_rl_20 = pretrain(
                X_test_2, y_test, X_train, y_train, RL_X, RL_y, n, 167,
                i_first, best_rf_params_RL, best_rf_params_real, RL=True
            )
        elif n == 67:
            accuracy_ol_50, sensitivity_ol_50, specificity_ol_50, roc_auc_ol_50 = pretrain(
                X_test_2, y_test, X_train, y_train, OL_X[simulated], OL_y, n, 133,
                i_first, best_rf_params_OL, best_rf_params_real
            )
            accuracy_rl_50, sensitivity_rl_50, specificity_rl_50, roc_auc_rl_50 = pretrain(
                X_test_2, y_test, X_train, y_train, RL_X, RL_y, n, 133,
                i_first, best_rf_params_RL, best_rf_params_real, RL=True
            )
        elif n == 100:
            accuracy_ol_100, sensitivity_ol_100, specificity_ol_100, roc_auc_ol_100 = pretrain(
                X_test_2, y_test, X_train, y_train, OL_X[simulated], OL_y, n, 100,
                i_first, best_rf_params_OL, best_rf_params_real
            )
            accuracy_rl_100, sensitivity_rl_100, specificity_rl_100, roc_auc_rl_100 = pretrain(
                X_test_2, y_test, X_train, y_train, RL_X, RL_y, n, 100,
                i_first, best_rf_params_RL, best_rf_params_real, RL=True
            )

    # final return with all values
    return (
        accuracy, sensitivity, specificity, roc_auc, threshold, importances_df,
        accuracy_ol_20, sensitivity_ol_20, specificity_ol_20, roc_auc_ol_20,
        accuracy_rl_20, sensitivity_rl_20, specificity_rl_20, roc_auc_rl_20,
        accuracy_ol_50, sensitivity_ol_50, specificity_ol_50, roc_auc_ol_50,
        accuracy_rl_50, sensitivity_rl_50, specificity_rl_50, roc_auc_rl_50,
        accuracy_ol_100, sensitivity_ol_100, specificity_ol_100, roc_auc_ol_100,
        accuracy_rl_100, sensitivity_rl_100, specificity_rl_100, roc_auc_rl_100, 
        X_test_len_sin, X_train_len_sin
    )


def create_output(outcomes):
    ## calculate mean accuracys for different approaches 
    # Initialize all result containers
    accuracys = []
    sensitivitys = []
    specificitys = []
    roc_auc_scores = []
    thresholds = []
    importances_list = []

    accuracys_OL_20 = []
    sensitivitys_OL_20 = []
    specificitys_OL_20 = []
    roc_auc_scores_OL_20 = []

    accuracys_RL_20 = []
    sensitivitys_RL_20 = []
    specificitys_RL_20 = []
    roc_auc_scores_RL_20 = []

    accuracys_OL_50 = []
    sensitivitys_OL_50 = []
    specificitys_OL_50 = []
    roc_auc_scores_OL_50 = []

    accuracys_RL_50 = []
    sensitivitys_RL_50 = []
    specificitys_RL_50 = []
    roc_auc_scores_RL_50 = []

    accuracys_OL_100 = []
    sensitivitys_OL_100 = []
    specificitys_OL_100 = []
    roc_auc_scores_OL_100 = []

    accuracys_RL_100 = []
    sensitivitys_RL_100 = []
    specificitys_RL_100 = []
    roc_auc_scores_RL_100 = []
    
    X_test_len = []
    X_train_len = []

    # Flatten outcomes into the above lists
    for sublist in outcomes:
        (acc, sens, spec, auc, thr, imp,
         acc_ol_20, sens_ol_20, spec_ol_20, auc_ol_20,
         acc_rl_20, sens_rl_20, spec_rl_20, auc_rl_20,
         acc_ol_50, sens_ol_50, spec_ol_50, auc_ol_50,
         acc_rl_50, sens_rl_50, spec_rl_50, auc_rl_50,
         acc_ol_100, sens_ol_100, spec_ol_100, auc_ol_100,
         acc_rl_100, sens_rl_100, spec_rl_100, auc_rl_100, 
         X_test_len_sin, X_train_len_sin) = sublist

        # baseline
        accuracys.append(acc)
        sensitivitys.append(sens)
        specificitys.append(spec)
        roc_auc_scores.append(auc)
        thresholds.append(thr)
        importances_list.append(imp)

        # OL 20
        accuracys_OL_20.append(acc_ol_20)
        sensitivitys_OL_20.append(sens_ol_20)
        specificitys_OL_20.append(spec_ol_20)
        roc_auc_scores_OL_20.append(auc_ol_20)

        # RL 20
        accuracys_RL_20.append(acc_rl_20)
        sensitivitys_RL_20.append(sens_rl_20)
        specificitys_RL_20.append(spec_rl_20)
        roc_auc_scores_RL_20.append(auc_rl_20)

        # OL 50
        accuracys_OL_50.append(acc_ol_50)
        sensitivitys_OL_50.append(sens_ol_50)
        specificitys_OL_50.append(spec_ol_50)
        roc_auc_scores_OL_50.append(auc_ol_50)

        # RL 50
        accuracys_RL_50.append(acc_rl_50)
        sensitivitys_RL_50.append(sens_rl_50)
        specificitys_RL_50.append(spec_rl_50)
        roc_auc_scores_RL_50.append(auc_rl_50)

        # OL 100
        accuracys_OL_100.append(acc_ol_100)
        sensitivitys_OL_100.append(sens_ol_100)
        specificitys_OL_100.append(spec_ol_100)
        roc_auc_scores_OL_100.append(auc_ol_100)

        # RL 100
        accuracys_RL_100.append(acc_rl_100)
        sensitivitys_RL_100.append(sens_rl_100)
        specificitys_RL_100.append(spec_rl_100)
        roc_auc_scores_RL_100.append(auc_rl_100)
        
        X_test_len.append(X_test_len_sin)
        X_train_len.append(X_train_len_sin)

    
    mean_r = sum(accuracys)/len(accuracys)
    std_r = np.std(accuracys)

    mean_accuracy_OL_20 = sum(accuracys_OL_20)/len(accuracys_OL_20)
    mean_accuracy_OL_50 = sum(accuracys_OL_50)/len(accuracys_OL_50)
    mean_accuracy_OL_100 = sum(accuracys_OL_100)/len(accuracys_OL_100)

    mean_accuracy_RL_20 = sum(accuracys_RL_20)/len(accuracys_RL_20)
    mean_accuracy_RL_50 = sum(accuracys_RL_50)/len(accuracys_RL_50)
    mean_accuracy_RL_100 = sum(accuracys_RL_100)/len(accuracys_RL_100)

    ## calculate p value for test of each approach vs standard forest

    n_test = np.mean(X_test_len)
    n_train = np.mean(X_train_len)

    n_test = round(n_test)

    sig_stat_OL_20, p_value_OL_20 = corrected_t_test(accuracys_OL_20, accuracys, n_train, n_test) ## adjust this values 
    sig_stat_OL_50, p_value_OL_50 = corrected_t_test(accuracys_OL_50, accuracys, n_train, n_test) ## adjust this values 
    sig_stat_OL_100, p_value_OL_100= corrected_t_test(accuracys_OL_100, accuracys, n_train, n_test) ## adjust this values 

    sig_stat_RL_20, p_value_RL_20 = corrected_t_test(accuracys_RL_20, accuracys, n_train, n_test) ## adjust this values 
    sig_stat_RL_50, p_value_RL_50 = corrected_t_test(accuracys_RL_50, accuracys, n_train, n_test) ## adjust this values 
    sig_stat_RL_100, p_value_RL_100 = corrected_t_test(accuracys_RL_100, accuracys, n_train, n_test) ## adjust this values 

    ## calculate mean sensitivitys and specificitys for different approaches

    sensitivity_mean = np.mean(sensitivitys)
    specificity_mean = np.mean(specificitys)

    sensitivity_mean_OL_20 = np.mean(sensitivitys_OL_20)
    sensitivity_mean_OL_50 = np.mean(sensitivitys_OL_50)
    sensitivity_mean_OL_100 = np.mean(sensitivitys_OL_100)
    specificity_mean_OL_20 = np.mean(specificitys_OL_20)
    specificity_mean_OL_50 = np.mean(specificitys_OL_50)
    specificity_mean_OL_100 = np.mean(specificitys_OL_100)

    sensitivity_mean_RL_20 = np.mean(sensitivitys_RL_20)
    sensitivity_mean_RL_50 = np.mean(sensitivitys_RL_50)
    sensitivity_mean_RL_100 = np.mean(sensitivitys_RL_100)
    specificity_mean_RL_20 = np.mean(specificitys_RL_20)
    specificity_mean_RL_50 = np.mean(specificitys_RL_50)
    specificity_mean_RL_100 = np.mean(specificitys_RL_100)

    ## calculate mean area under the curve

    mean_roc_auc = np.mean(roc_auc_scores)

    mean_roc_auc_OL_20 = np.mean(roc_auc_scores_OL_20)
    mean_roc_auc_OL_50 = np.mean(roc_auc_scores_OL_50)
    mean_roc_auc_OL_100 = np.mean(roc_auc_scores_OL_100)

    mean_roc_auc_RL_20 = np.mean(roc_auc_scores_RL_20)
    mean_roc_auc_RL_50 = np.mean(roc_auc_scores_RL_50)
    mean_roc_auc_RL_100 = np.mean(roc_auc_scores_RL_100)

    ### now standard deviations
    
    # Accuracy - OL
    std_accuracy_OL_20  = np.std(accuracys_OL_20)
    std_accuracy_OL_50  = np.std(accuracys_OL_50)
    std_accuracy_OL_100 = np.std(accuracys_OL_100)

    # Accuracy - RL
    std_accuracy_RL_20  = np.std(accuracys_RL_20)
    std_accuracy_RL_50  = np.std(accuracys_RL_50)
    std_accuracy_RL_100 = np.std(accuracys_RL_100)

    # Sensitivity and Specificity (overall)
    sensitivity_std  = np.std(sensitivitys)
    specificity_std  = np.std(specificitys)

    # Sensitivity - OL
    sensitivity_std_OL_20  = np.std(sensitivitys_OL_20)
    sensitivity_std_OL_50  = np.std(sensitivitys_OL_50)
    sensitivity_std_OL_100 = np.std(sensitivitys_OL_100)

    # Specificity - OL
    specificity_std_OL_20  = np.std(specificitys_OL_20)
    specificity_std_OL_50  = np.std(specificitys_OL_50)
    specificity_std_OL_100 = np.std(specificitys_OL_100)

    # Sensitivity - RL
    sensitivity_std_RL_20  = np.std(sensitivitys_RL_20)
    sensitivity_std_RL_50  = np.std(sensitivitys_RL_50)
    sensitivity_std_RL_100 = np.std(sensitivitys_RL_100)

    # Specificity - RL
    specificity_std_RL_20  = np.std(specificitys_RL_20)
    specificity_std_RL_50  = np.std(specificitys_RL_50)
    specificity_std_RL_100 = np.std(specificitys_RL_100)

    # ROC AUC - overall
    std_roc_auc  = np.std(roc_auc_scores)

    # ROC AUC - OL
    std_roc_auc_OL_20  = np.std(roc_auc_scores_OL_20)
    std_roc_auc_OL_50  = np.std(roc_auc_scores_OL_50)
    std_roc_auc_OL_100 = np.std(roc_auc_scores_OL_100)

    # ROC AUC - RL
    std_roc_auc_RL_20  = np.std(roc_auc_scores_RL_20)
    std_roc_auc_RL_50  = np.std(roc_auc_scores_RL_50)
    std_roc_auc_RL_100 = np.std(roc_auc_scores_RL_100)
    
    results = pd.DataFrame({
        "Balanced Accuracy (mean)": [
            mean_r, mean_accuracy_OL_20, mean_accuracy_OL_50, mean_accuracy_OL_100,
            mean_accuracy_RL_20, mean_accuracy_RL_50, mean_accuracy_RL_100
        ],
        "Balanced Accuracy (std)": [
            std_r, std_accuracy_OL_20, std_accuracy_OL_50, std_accuracy_OL_100,
            std_accuracy_RL_20, std_accuracy_RL_50, std_accuracy_RL_100
        ],

        "AUROC (mean)": [
            mean_roc_auc, mean_roc_auc_OL_20, mean_roc_auc_OL_50, mean_roc_auc_OL_100,
            mean_roc_auc_RL_20, mean_roc_auc_RL_50, mean_roc_auc_RL_100
        ],
        "AUROC (std)": [
            std_roc_auc, std_roc_auc_OL_20, std_roc_auc_OL_50, std_roc_auc_OL_100,
            std_roc_auc_RL_20, std_roc_auc_RL_50, std_roc_auc_RL_100
        ],

        "Sensitivity (mean)": [
            sensitivity_mean, sensitivity_mean_OL_20, sensitivity_mean_OL_50, sensitivity_mean_OL_100,
            sensitivity_mean_RL_20, sensitivity_mean_RL_50, sensitivity_mean_RL_100
        ],
        "Sensitivity (std)": [
            sensitivity_std, sensitivity_std_OL_20, sensitivity_std_OL_50, sensitivity_std_OL_100,
            sensitivity_std_RL_20, sensitivity_std_RL_50, sensitivity_std_RL_100
        ],

        "Specificity (mean)": [
            specificity_mean, specificity_mean_OL_20, specificity_mean_OL_50, specificity_mean_OL_100,
            specificity_mean_RL_20, specificity_mean_RL_50, specificity_mean_RL_100
        ],
        "Specificity (std)": [
            specificity_std, specificity_std_OL_20, specificity_std_OL_50, specificity_std_OL_100,
            specificity_std_RL_20, specificity_std_RL_50, specificity_std_RL_100
        ],

        "p_value_standard": [
            "H0", p_value_OL_20, p_value_OL_50, p_value_OL_100,
            p_value_RL_20, p_value_RL_50, p_value_RL_100
        ],
        "t_value_standard": [
            "None", sig_stat_OL_20, sig_stat_OL_50, sig_stat_OL_100,
            sig_stat_RL_20, sig_stat_RL_50, sig_stat_RL_100
        ],
        "iterations": [
            i, i, i, i, i, i, i
        ]
    }, index=[
        "Standard", "OL_20", "OL_50", "OL_100", "RL_20", "RL_50", "RL_100"
    ])




    results.to_excel("MaRep/results/results_rem_smote.xlsx")



pool = Pool(25)
runs_list = []
outcomes = []
for i in range (number_iterations):
    runs_list.append(i)
outcomes[:] = pool.map(perform_one_iteration,runs_list)
create_output(outcomes)

