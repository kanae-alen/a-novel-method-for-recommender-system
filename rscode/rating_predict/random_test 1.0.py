import numpy as np
import pandas as pd
import math
import copy
R = np.array([
              [1, 2,      1,      np.nan,      3,      3],
              [3,      np.nan,      4,      5, np.nan,      5     ],
              [1,      4, 5,     np.nan,      3,      2],
              [1, np.nan,      4, 2,      5,      np.nan     ],
])

# lambda_new = 0.02

# regularization_u_split = 0
# regularization_i_split = 0
# regularization = 0
# S=[]
# S1=[]

mu = np.nansum(R) / np.sum(~np.isnan(R))
bu = (np.random.rand(R.shape[0]) - 0.5) * 0.1
bi = (np.random.rand(R.shape[1]) - 0.5) * 0.1
pu = (np.random.rand(R.shape[0] ,2) - 0.5) * 0.1
qi = (np.random.rand(R.shape[1] ,2) - 0.5) * 0.1
S = (np.random.rand(2 ,2) - 0.5) * 0.1

a = [1,2,3]
b = [1,2,3]
error_matrix = R - (mu + np.dot(np.dot(pu, S), qi.T) + np.matrix(bu).T + bi)
A =   -S[:,0].T
B = np.dot(pu[0 ,:], S[:,0])
# for u in range(R.shape[0]):
#      regularization_u_split = (~np.isnan(R[u,:])).sum()
#      S.append(regularization_u_split)

# for i in range(R.shape[1]):
#      regularization_i_split = (~np.isnan(R[:,i])).sum() 
#      S1.append(regularization_i_split)

# print(mu)
# print(bu)
# print(bi)
# print(pu)
# print(pu[0])
# print(S)
# print(S[0])
# print(qi)
# print(qi[0])

# print(regularization_u_split)
# print(regularization_i_split)
# print(regularization)
# print('pu[0] = {}'.format(pu[0]))
# # print('bu = {}'.format(bu))
# x = np.sum(np.square(pu[0]))
# print('error_matrix = {}'.format(error_matrix))
# print('S = {}'.format(S))
# print('A = {}'.format(A))
print('B = {}'.format(B))