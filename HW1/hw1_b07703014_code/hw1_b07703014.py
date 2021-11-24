'''
Financial Technology
Assignment 1
B07703014 財金三 蔡承翰
Due  24-10-2020
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as pps


def solve_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr, regulate = False, Lambda = 0, is_bias = False):
    '''Solve regular linear regression without bias'''

    if not regulate:
        # Normal equation w = (X'X)^(-1)X'Y
        betas_arr = np.linalg.pinv(train_x_arr.T @ train_x_arr) @ train_x_arr.T @ train_y_arr
    else:
        # Minmize J: w = (X'X/n+lambda*I)^(-1)X'Y/n
        over_n = 1/np.size(train_x_arr, axis = 0)
        pinv_J = np.linalg.pinv((over_n * (train_x_arr.T @ train_x_arr)) + (Lambda * np.eye(np.size(train_x_arr, axis = 1))))
        betas_arr = ((pinv_J @ train_x_arr.T) @ train_y_arr) * over_n

    # Compute the error
    if is_bias:
        bias = np.mean(train_y_arr - (train_x_arr @ betas_arr))
        predict_lin_reg = test_x_arr @ betas_arr + bias
    else:
        predict_lin_reg = test_x_arr @ betas_arr
    
    error_arr = test_y_arr - predict_lin_reg  # No bias
    RMSE = np.sqrt(np.mean(error_arr ** 2))
    return predict_lin_reg, RMSE


def solve_Baysian_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr, alpha = 1.0):
    '''Baysian linear regression, only need to compute the mean of the postier distribution'''
    # Goal. Find mu_m = sigma_m(X'Y+sigma_0^(-1)mu_0), where sigma_m = (X'X + sigma_0^(-1))^(-1)
    mu_0 = 0
    sigma_0 = (1/alpha) * np.eye(np.size(train_x_arr, axis = 1))

    mu_0 = [mu_0 for i in range(np.size(train_x_arr, axis = 1))]
    mu_0 = np.expand_dims(mu_0, axis = 1)

    # Compute sigma_m
    sigma_m = (train_x_arr.T @ train_x_arr) + np.linalg.inv(sigma_0)
    sigma_m = np.linalg.inv(sigma_m)
    
    # Compute mu_m
    mu_m = sigma_m @ ((train_x_arr.T @ train_y_arr) + (np.linalg.inv(sigma_0) @ mu_0))
    # mu_m = np.linalg.inv(train_x_arr.T @ train_x_arr + (alpha * np.eye(np.size(train_x_arr, axis = 1)))) @ train_x_arr.T @ train_y_arr
    
    # Compute RMSE
    bias_B = np.mean(train_y_arr - (train_x_arr @ mu_m))
    predict_lin_reg_bay = test_x_arr @ mu_m + bias_B
    error_arr_B = test_y_arr - predict_lin_reg_bay
    RMSE_B = np.sqrt(np.mean(error_arr_B ** 2))
    return predict_lin_reg_bay, RMSE_B, mu_m, bias_B


def main():
    
    # 1.(a)
    train_df = pps.read_data("../train.csv")
    testset_df, trainset_df = pps.divide_data(train_df)
    norm_base_df = trainset_df.drop("G3", axis = 1)  # normalization standard
    train_y_arr, train_x_arr = pps.preprocess(trainset_df, norm_base_df)
    test_y_arr, test_x_arr = pps.preprocess(testset_df, norm_base_df)
    print(train_x_arr.shape)
    print(train_y_arr.shape)

    # 1.(b)
    # Normal equation: w = (X'X)^(-1)X'Y
    predict_lin_reg, RMSE = solve_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr)
    # Output
    print("Problem 1.(b)")
    print("RMSE without the bias:")
    print(RMSE)
    print()
    
    # 1.(c) Regression with regularization
    predict_regular, RMSE_J = solve_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr, regulate = True, Lambda = (1/2))
    # Output
    print("Problem 1.(c)")
    print("RMSE with regularization without bias:")
    print(RMSE_J)
    print()

    # 1.(d) Bias in (c)
    predict_lin_reg_regular_bias, RMSE_Jb = solve_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr, regulate = True, Lambda = (1/2), is_bias = True)
    # Output
    print("Problem 1.(d)")
    print("RMSE with regularization and bias:")
    print(RMSE_Jb)
    print()
    
    # (1).(e) Bayesian linear regression
    predict_lin_reg_bay, RMSE_B, mu_m, bias = solve_Baysian_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr)
    # Output
    print("Problem 1.(e)")
    print("RMSE with Baysian linear regression and bias:")
    print(RMSE_B)
    print()

    # (1).(f) Plot the actual y and predictions from above
    idx_list = [(i+1) for i in range(np.size(test_y_arr))]
    plt.plot(idx_list, test_y_arr, label = "Ground Truth")
    plt.plot(idx_list, predict_lin_reg, label = "(11.60)Linear Regression", linewidth = 3)
    plt.plot(idx_list, predict_regular, label = "(11.59)Linear Regression (reg)")
    plt.plot(idx_list, predict_lin_reg_regular_bias, label = "(3.39)Linear Regression (r/b)")
    plt.plot(idx_list, predict_lin_reg_bay, label = "(3.43)Baysian Linear Regression")
    plt.legend()
    plt.show()
    
    # 1.(g) Baysian linear regression prediction on test_no_G3
    test_no_G3_df = pps.read_data("../test_no_G3.csv", no_G3 = True)
    test_no_G3_x_arr = pps.preprocess(test_no_G3_df.drop("ID", axis = 1), norm_base_df, no_G3 = True)
    tried_alpha = []
    tried_RMSE = []
    
    # Same as (e), initialization
    alpha_g = 1
    predict_arr = [0 for i in range(np.size(test_no_G3_x_arr, axis = 0))]
    predict_arr = np.expand_dims(predict_arr, axis = 1)
    RMSE_g = 9999

    for attempt_alpha in np.linspace(0.1, 1000, 100):
        predict, RMSE_B, mu_m, bias = solve_Baysian_linear_reg(train_x_arr, train_y_arr, test_x_arr, test_y_arr, attempt_alpha)
        tried_alpha.append(attempt_alpha)
        tried_RMSE.append(RMSE_B)
        if RMSE_B < RMSE_g:
            RMSE_g = RMSE_B
            predict_arr = test_no_G3_x_arr @ mu_m + bias
            alpha_g = attempt_alpha
    
    # Write into txt
    output_arr = np.concatenate((np.expand_dims(test_no_G3_df["ID"].to_numpy(), axis = 1), predict_arr), axis = 1)
    np.savetxt("../B07703014_1.txt", output_arr, fmt = ["%d", "%1.1f"])

    # Maybe a graph
    plt.plot(tried_alpha, tried_RMSE)
    plt.xlabel("alpha")
    plt.ylabel("RMSE")
    plt.show()
    
    print("Problem 1.(g)")
    print("Best alpha: "+str(alpha_g))
    print("RMSE: "+str(RMSE_g))
    print()
    print()

    # 2.(a)
    print("Problem 2.")
    print()
    train_data_x, train_target_y, test_data_x, test_target_y = pps.preprocess_adult()

    predict_lin_reg_2, RMSE_lin_reg_2 = solve_linear_reg(train_data_x, train_target_y, test_data_x, test_target_y)
    print("RMSE without the bias:")
    print(RMSE_lin_reg_2)
    print()

    predict_regular_2, RMSE_J_2 = solve_linear_reg(train_data_x, train_target_y, test_data_x, test_target_y, regulate = True, Lambda = (1/2))
    # Output
    print("RMSE with regularization without bias:")
    print(RMSE_J_2)
    print()

    predict_lin_reg_regular_bias_2, RMSE_Jb_2 = solve_linear_reg(train_data_x, train_target_y, test_data_x, test_target_y, regulate = True, Lambda = (1/2), is_bias = True)
    # Output
    print("RMSE with regularization and bias:")
    print(RMSE_Jb_2)
    print()

    alpha_g_2 = 1
    predict_bay_2 = []
    RMSE_g_2 = 9999
    tried_alpha_2 = []
    tried_RMSE_2 = []

    for attempt_alpha in np.linspace(1500, 2500, 100):
        predict, RMSE_B, mu_m, bias = solve_Baysian_linear_reg(train_data_x, train_target_y, test_data_x, test_target_y, attempt_alpha)
        tried_alpha_2.append(attempt_alpha)
        tried_RMSE_2.append(RMSE_B)
        if RMSE_B < RMSE_g_2:
            RMSE_g_2 = RMSE_B
            predict_bay_2 = predict
            alpha_g_2 = attempt_alpha
    
    print("RMSE with Bayesian linear regression, tunable alpha and bias:")
    print(RMSE_g_2)
    print("Best alpha:")
    print(alpha_g_2)

    plt.plot(tried_alpha_2, tried_RMSE_2)
    plt.xlabel("alpha")
    plt.ylabel("RMSE")
    plt.show()

    idx_list = [(i+1) for i in range(np.size(test_target_y))]
    plt.plot(idx_list, test_target_y, label = "Ground Truth")
    plt.plot(idx_list, predict_lin_reg_2, label = "(0.41)Linear Regression")
    plt.plot(idx_list, predict_regular_2, label = "(0.42)Linear Regression (reg)", linewidth = 3)
    plt.plot(idx_list, predict_lin_reg_regular_bias_2, label = "(0.34)Linear Regression (r/b)")
    plt.plot(idx_list, predict_bay_2, label = "(0.34)Baysian Linear Regression", linewidth = 3)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()


