import numpy as np
import pandas as pd

def read_data(file_name, no_G3 = False):
    '''
    Predictors: *school, *sex, age, *famsize, studytime, failures, *activities, *higher, *internet, 
    *romantic, famrel, freetime, goout, Dalc, Walc, health, absences (*are dummy variables)
    '''
    df = pd.read_csv(file_name)
    # Extract predictors and target
    if no_G3:
        df = df[["ID", "school", "sex", "age", "famsize", "studytime", "failures", "activities", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]]
    else:
        df = df[["school", "sex", "age", "famsize", "studytime", "failures", "activities", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G3"]]  
    df = pd.get_dummies(df)  # transform binary columns to one-hot encoding vectors
    return df

def divide_data(df):
    '''Divide the train data into trainset and testset'''
    import random as rand
    rand.seed(703014)
    row_count = len(df.index)
    testset_count = int(row_count * 0.2)
    testset_idxes = rand.sample(list(df.index), testset_count)  # Choose testset psuedo-ramdomly with fixed seed
    testset_idxes = sorted(testset_idxes)
    testset_df = df.iloc[testset_idxes]
    trainset_df = df.drop(testset_idxes)
    return testset_df, trainset_df

def normalize(df, norm_df):
    '''Column-wise normalization by mean and std from trainset_df'''
    return (df-norm_df.mean())/(norm_df.std()+1E-8)  # In case that std is zero

def extract_target(set_df, norm_df):
    '''Extract vector G3, return target and predictor'''
    target_df = set_df["G3"]
    set_df = set_df.drop("G3", axis = 1)
    set_df = normalize(set_df, norm_df)
    return np.expand_dims(target_df.to_numpy(), axis = 1), set_df.to_numpy()

def preprocess(set_df, norm_df, no_G3 = False):
    '''normalize, extract target and return narray'''
    if no_G3:
        factor_arr = normalize(set_df, norm_df)
        return factor_arr.to_numpy()
    else:
        target_arr, predict_arr = extract_target(set_df, norm_df)
        return target_arr, predict_arr

def preprocess_adult():
    '''read, normalize, extract target and return narray for (2)'''
    adult_data_df = pd.read_csv("../adult.data", header = None)
    adult_test_df = pd.read_csv("../adult.test", header = None, skiprows=[0])

    adult_data_df = pd.get_dummies(adult_data_df)
    adult_test_df = pd.get_dummies(adult_test_df)

    train_target_df = adult_data_df[adult_data_df.columns[-1]].to_numpy()  # >50K = 1, <=50K = 0
    train_target_arr = np.expand_dims(train_target_df, axis = 1)
    test_target_df = adult_test_df[adult_test_df.columns[-1]].to_numpy()
    test_target_arr = np.expand_dims(test_target_df, axis = 1)

    common_column = [col for col in adult_data_df.columns if col in adult_test_df.columns]
    adult_data_df = adult_data_df[common_column]
    adult_test_df = adult_test_df[common_column]
    
    # normalize
    adult_data_arr = adult_data_df.to_numpy()
    adult_test_arr = adult_test_df.to_numpy()
    adult_test_arr = (adult_test_arr - np.mean(adult_data_arr, axis = 0)) / np.std(adult_data_arr, axis = 0)
    adult_data_arr = (adult_data_arr - np.mean(adult_data_arr, axis = 0)) / np.std(adult_data_arr, axis = 0)

    return adult_data_arr, train_target_arr, adult_test_arr, test_target_arr