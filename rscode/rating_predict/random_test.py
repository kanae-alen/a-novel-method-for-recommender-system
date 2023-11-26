import numpy as np
import pandas as pd
import math
np.set_printoptions(precision=3)



# k为潜在因子模型矩阵的维数，即用户特征矩阵和物品特征矩阵的列的数目
k = 2

# 设置循环的次数，一次循环会遍历一遍数据
epochs = 20

# 设置正则化参数
λ = 0.02

# 设置学习率
gamma = 0.02

# 设置批量梯度下降一次抽取的样本数目(暂时不做)
batch_size = 100

# 评价算法精确度的评分误差，先初始化为0
RMSE = 0

# # 读取训练集数据，并且生成一个与数据维度相同的空值矩阵R
# #——————————————————————————————————————————————————————————————————
df_u1_train = pd.read_csv('u1.base', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])
shape = (df_u1_train.max().loc['user_id'], df_u1_train.max().loc['item_id'])
R = np.nan * np.ones(shape) 
# #——————————————————————————————————————————————————————————————————

# # # 测试用的R矩阵
# R = np.array([
#               [1, 2,      1,      np.nan,      3,      3],
#               [3,      np.nan,      4,      5, np.nan,      5     ],
#               [1,      4, 5,     np.nan,      3,      2],
#               [1, np.nan,      4, 2,      5,      np.nan     ],
# ])

# 将数据中的元素一个一个赋值给R矩阵，形成完整的评分矩阵，没有评分数据的地方仍未nan
#——————————————————————————————————————————————————————————————————
for i in df_u1_train.index:
   row = df_u1_train.loc[i]
   R[row['user_id'] -1 , row['item_id'] - 1] = row['rating']
#——————————————————————————————————————————————————————————————————

# 用于数据初始化的函数, mu为矩阵全体平均值，bu为用户偏差，bi为物品偏差
# pu为用户特征矩阵，qi为物品特征矩阵
def get_initial_values(R):
    # 各行列またはベクトルの値を、-0.05〜+0.05の乱数で初期化
    mu = np.nansum(R) / np.sum(~np.isnan(R))
    bu = (np.random.rand(R.shape[0]) - 0.5) * 0.1
    bi = (np.random.rand(R.shape[1]) - 0.5) * 0.1
    pu = (np.random.rand(R.shape[0] ,k) - 0.5) * 0.1
    qi = (np.random.rand(R.shape[1] ,k) - 0.5) * 0.1
    
    return (mu, bu, bi, pu, qi)

# 用于计算误差矩阵，原矩阵缺失的地方不会参与计算，仍然为nan值
def get_error_matrix(R, mu, bu, bi, pu, qi):
    # 誤差eの計算
    error_matrix = R - (mu + np.dot(pu, qi.T) + np.matrix(bu).T + bi)
    return error_matrix


# 计算误差函数的值
def error_function(R, mu, bu, bi, pu, qi, λ):
    # 誤差に正則化制約を加えた目的関数を定義
    error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)
    error_main = (np.sum(np.square(error_matrix[ ~np.isnan(error_matrix)]))) / 2
    regularization = (λ * (np.sum(np.square(bu)) + np.sum(np.square(bi)) + np.sum(np.square(pu)) + np.sum(np.square(qi)))) / 2
    
    return error_main + regularization

def sgd(R, epochs, λ, gamma, batch_size):

    # 確率的勾配法で分解した行列とバイアスを求める
    mu, bu, bi, pu, qi = get_initial_values(R)
    error_list = []
    error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)
    u_index, i_index = np.where(~np.isnan(R))
    rating_count = len(u_index)
    length_u = R.shape[0]
    length_i = R.shape[1]
    targets = np.arange(rating_count)
    batch_count = math.ceil(rating_count / batch_size)

    for epoch in range(epochs):
        
        # 欠測値を除いた評価値のインデックスを取得してランダムに並び替える
        np.random.shuffle(targets)
        for batch_base in range(batch_count):
        # ミニバッチ単位で誤差の計算とパラメータの更新を行う
            
            delta_bu = np.zeros_like(bu)
            delta_bi = np.zeros_like(bi)
            delta_pu = np.zeros_like(pu)
            delta_qi = np.zeros_like(qi)
            u_index_list = []
            i_index_list = []
            for batch_offset in range(batch_size):

                target_index = batch_size * batch_base + batch_offset
                u_index_random = u_index[targets[target_index]]
                i_index_random = i_index[targets[target_index]]
                u_index_list.append(u_index_random)
                i_index_list.append(i_index_random)
                e_ui = error_matrix[u_index_random, i_index_random]
                
                # パラメータの更新値を累積
                delta_bu[u_index_random] += gamma * e_ui 
                delta_bi[i_index_random] += gamma * e_ui 
                delta_pu[u_index_random ,:] += gamma * e_ui * qi[i_index_random ,:] 
                delta_qi[i_index_random ,:] += gamma * e_ui * pu[u_index_random ,:] 
            
            u_index_list = list(set(u_index_list))
            i_index_list = list(set(i_index_list))
            for u in u_index_list:
                delta_bu[u] = delta_bu[u] - gamma *  λ * bu[u]
                delta_pu[u ,:] = delta_pu[u ,:] - gamma *  λ * pu[u ,:]
            for i in i_index_list:
                delta_bi[i] = delta_bi[i] - gamma *  λ * bi[i]
                delta_qi[i] = delta_qi[i] - gamma *  λ * qi[i]
            # パラメータの更新
            bu += delta_bu
            bi += delta_bi
            pu += delta_pu
            qi += delta_qi

            error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)
        error = error_function(R, mu, bu, bi, pu, qi, λ)
        error_list.append(error)     
  # 誤差の計算
    # error = error_function(R, mu, bu, bi, pu, qi, λ)
    # print('Error: {}'.format(error))
    
    # 予測した評価値を生成
    expected = mu + np.dot(pu, qi.T) + np.matrix(bu).T + bi
    return expected, error_list


expected, error_list = sgd(R, epochs, λ, gamma, batch_size)
df_u1_test = pd.read_csv('u1.test', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])
for i in df_u1_test.index:
   row = df_u1_test.loc[i]
   RMSE += math.pow( expected[row['user_id'] -1 , row['item_id'] - 1] - row['rating'] , 2 )
RMSE = math.sqrt(RMSE/len(df_u1_test.index))
# print('expected = {}'.format(expected))
print('RMSE = {}'.format(RMSE))
print('error_list = {}'.format(error_list))
