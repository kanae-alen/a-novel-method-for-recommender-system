import numpy as np
import pandas as pd
import math
import copy
np.set_printoptions(precision=3)

# 读取训练集数据，并且生成一个与数据维度相同的空值矩阵R
#——————————————————————————————————————————————————————————————————
df_train = pd.read_csv('D:/rscode/matrix_predict_tradeoff_mmr/train_data.txt', sep='\t', names=['user_id','item_id', 'rating'])
shape = (df_train.max().loc['user_id'], df_train.max().loc['item_id'])
R = np.nan * np.ones(shape) 
#——————————————————————————————————————————————————————————————————

# 读取测试集数据，并且生成一个与数据维度相同的空值矩阵R_test
#——————————————————————————————————————————————————————————————————
df_test = pd.read_csv('D:/rscode/matrix_predict_tradeoff_mmr/test_data.txt', sep='\t', names=['user_id','item_id', 'rating'])
shape_test = (df_test.max().loc['user_id'], df_test.max().loc['item_id'])
R_test = np.nan * np.ones(shape_test) 
#——————————————————————————————————————————————————————————————————

# 将数据中的元素一个一个赋值给R矩阵，形成完整的训练矩阵，没有评分数据的地方仍未nan
#——————————————————————————————————————————————————————————————————
for i in df_train.index:
   row = df_train.loc[i]
   R[row['user_id'] -1 , row['item_id'] - 1] = row['rating']
#——————————————————————————————————————————————————————————————————

# 将数据中的元素一个一个赋值给R_test矩阵，形成完整的测试矩阵，没有评分数据的地方仍未nan
#——————————————————————————————————————————————————————————————————
for i in df_test.index:
   row = df_test.loc[i]
   R_test[row['user_id'] -1 , row['item_id'] - 1] = row['rating']
#——————————————————————————————————————————————————————————————————

# k为潜在因子模型矩阵的维数，即用户特征矩阵和物品特征矩阵的列的数目
k = 2

# 设置循环的次数，一次循环会遍历一遍数据
epochs = 20

# 设置正则化参数
lambda_new = 0.02

# 设置学习率
gamma = 0.02

# 设置批量梯度下降一次抽取的样本数目(暂时不做)
batch_size = 103

# 评价算法精确度的评分误差，先初始化为0
RMSE = 0

# 评价两个物品的共同用户的数目阈值(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
β_u = 20
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 高评价的阈值(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
threshold = 3.5
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 筛选分数前topN1(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
topN1 = 50
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 生成用户索引集合
U = np.arange(R.shape[0])
#print('U = {}'.format(U))

# 生成物品索引集合
I = np.arange(R.shape[1])
#print('I = {}'.format(I))

# 用户的总人数
total_users = len(U)

# 物品的总数目
total_items = len(I)

# 生成对应用户评价过的物品索引集合
#——————————————————————————————————————————————————————————————————
# 想看具体细节的话可以用下面这段代码
#for u in U:
#    print('I{} = {}'.format(u, I[~np.isnan(R)[u,:]]))
#——————————————————————————————————————————————————————————————————
Iu = [I[~np.isnan(R)[u,:]] for u in U]
#print('Iu = ')
#pprint.pprint(Iu)

# 生成评价过对应物品的用户索引集合
Ui = [U[~np.isnan(R)[:,i]] for i in I]
#print('Ui = ')
#pprint.pprint(Ui)

# 生成用户未评价过的物品索引集合
Iu_not = [I[np.isnan(R)[u,:]] for u in U]
#print('Iu_not = ')
#pprint.pprint(Iu_not)

# # # 测试用的R矩阵
# R = np.array([
#               [1, 2,      1,      np.nan,      3,      3],
#               [3,      np.nan,      4,      5, np.nan,      5     ],
#               [1,      4, 5,     np.nan,      3,      2],
#               [1, np.nan,      4, 2,      5,      np.nan     ],
# ])

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
    error_matrix = R - (mu + np.dot(pu, qi.T) + np.matrix(bu).T + bi)
    return error_matrix


# 计算误差函数的值,分为两部分，首先计算误差矩阵内元素的平方和，然后加上正则化项
def error_function(R, mu, bu, bi, pu, qi, lambda_new):
    # 誤差に正則化制約を加えた目的関数を定義
    error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)
    error_main = (np.sum(np.square(error_matrix[ ~np.isnan(error_matrix)]))) / 2

    regularization_u_split = 0
    regularization_i_split = 0
    regularization = 0

    for u in range(R.shape[0]):
        regularization_u_split += lambda_new * (~np.isnan(R[u,:])).sum() * ( np.square(bu[u]) + np.sum(np.square(pu[u])) ) / 2
        
    for i in range(R.shape[1]):
        regularization_i_split += lambda_new * (~np.isnan(R[:,i])).sum() * ( np.square(bi[i]) + np.sum(np.square(qi[i])) ) / 2

    regularization = regularization_u_split + regularization_i_split
    
    return error_main + regularization

# 批量随机梯度下降法求解
def msgd(R, epochs, lambda_new, gamma, batch_size):

    # 首先初始化各参数，使用数据初始化的函数
    mu, bu, bi, pu, qi = get_initial_values(R)

    # 用于存储误差的列表，首先置为空
    error_list = []

    # 生成误差矩阵，对应后续使用的eui
    error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)

    # 将评分不为0的（u，i）索引（即rui不为0）中的u赋值给u_index , i赋值给i_index
    u_index, i_index = np.where(~np.isnan(R))

    # 计算R矩阵中实际评分不为0的元素个数，便于后续批量计算
    rating_count = len(u_index)

    # 生成实际评分不为0的元素的索引列表，如有5个元素则生成列表0，1，2，3，4
    targets = np.arange(rating_count)

    # 批量输入计算的轮次，如共有80000个数据，一次批量输入100个，则需要计算800次
    batch_count = math.ceil(rating_count / batch_size)

    # epoch这里是对应要循环计算多少次数据，如为20，则会遍历计算20次这80000个数据
    for epoch in range(epochs):
        
        # 每次遍历计算完一次数据后，都会将target给重新打乱，如原本顺序为0，1，2，3，4，打乱后可能变为2，4，0，1，3
        np.random.shuffle(targets)

        # 这里是一次遍历计算数据所需要的轮次，前面已经计算过了
        for batch_base in range(batch_count):
            
            # 这四个参数用于暂时存放需要更新的参数值，暂时置为0
            delta_bu = np.zeros_like(bu)
            delta_bi = np.zeros_like(bi)
            delta_pu = np.zeros_like(pu)
            delta_qi = np.zeros_like(qi)

            # 这是最后一层循环，一次输入一定量的数据来更新一次参数
            for batch_offset in range(batch_size):
                
                # 这三个代码用于顺序抽取批量的数据，获得它们对应的索引
                target_index = batch_size * batch_base + batch_offset
                u_index_random = u_index[targets[target_index]]
                i_index_random = i_index[targets[target_index]]

                # 获取误差矩阵对应的eui，用于参数更新
                e_ui = error_matrix[u_index_random, i_index_random]
                
                # 更新参数，但是还有一部分参数由于代码问题还未更新完
                delta_bu[u_index_random] += gamma * ( e_ui -  lambda_new * bu[u_index_random])
                delta_bi[i_index_random] += gamma * ( e_ui -  lambda_new * bi[i_index_random])
                delta_pu[u_index_random ,:] += gamma * ( e_ui * qi[i_index_random ,:] - lambda_new * pu[u_index_random ,:] )
                delta_qi[i_index_random ,:] += gamma * ( e_ui * pu[u_index_random ,:] - lambda_new * qi[i_index_random ,:] )
            
            # 完成参数的更新
            bu += delta_bu
            bi += delta_bi
            pu += delta_pu
            qi += delta_qi

            # 完成误差矩阵的更新
            error_matrix = get_error_matrix(R, mu, bu, bi, pu, qi)
        
        # 每次遍历计算完成一轮数据，计算一遍误差
        error = error_function(R, mu, bu, bi, pu, qi, lambda_new)

        # 将误差记录下来，放到error_list里面
        error_list.append(error)     
  
    
    # 当算法完成后，生成完整的预测评分矩阵，此时该矩阵既有预测评分值，也有原本就有的分值
    expected = mu + np.dot(pu, qi.T) + np.matrix(bu).T + bi

    return expected, error_list

# 将评分矩阵和误差分为传给对应的参数
expected, error_list = msgd(R, epochs, lambda_new, gamma, batch_size)


#生成一个只有预测评分的矩阵
R_predict = {}

for u in U:
    R_predict[u] = {i: expected[u, i] for i in Iu_not[u]}

# 旧的提案
R_topN1_predict = {u: dict(sorted(R_predict[u].items(), key=lambda x:x[1], reverse=True)[:topN1]) for u in U}











# 把expected里面的预测的值与原来的值做一个对比
# expected_observed = copy.deepcopy(R)




# for i in df_u1_train.index:
#    row = df_u1_train.loc[i]
#    expected_observed[row['user_id'] -1 , row['item_id'] - 1] = expected[row['user_id'] -1 , row['item_id'] - 1]

# df_u1_test = pd.read_csv('u1.test', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])

# for i in df_u1_test.index:
#    row = df_u1_test.loc[i]
#    RMSE += math.pow( expected[row['user_id'] -1 , row['item_id'] - 1] - row['rating'] , 2 )

# RMSE = math.sqrt(RMSE/len(df_u1_test.index))

# print('RMSE = {}'.format(RMSE))
# print('k = {}'.format(k))
# print('lambda_new = {}'.format(lambda_new))
# print('gamma = {}'.format(gamma))
# print('batch_size = {}'.format(batch_size))
# print('error_list = {}'.format(error_list))