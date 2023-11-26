import numpy as np
import pandas as pd
import pprint
import math
np.set_printoptions(precision=3)

# 读取训练集数据，并且生成一个与数据维度相同的空值矩阵R
#——————————————————————————————————————————————————————————————————
df_u1_train = pd.read_csv('u1.base', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])
shape = (df_u1_train.max().loc['user_id'], df_u1_train.max().loc['item_id'])
R = np.nan * np.ones(shape) 
#——————————————————————————————————————————————————————————————————

# 测试用的R矩阵
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

# 选取的最大相似度邻居数目
K_USERS = 40

# 相似度的阈值
THETA = 0

# 评价两个物品的共同用户的数目阈值
β_u = 40

# 评价算法精确度的评分误差，先初始化为0
RMSE = 0

# 生成用户索引集合
U = np.arange(R.shape[0])
#print('U = {}'.format(U))


# 生成物品索引集合
I = np.arange(R.shape[1])
#print('I = {}'.format(I))

# 计算评分矩阵的稀疏度
#print('Rの全要素数 = {}'.format(R.size))
#print('観測値 = \n{}'.format(np.count_nonzero(~np.isnan(R))))
sparsity = 1 - np.count_nonzero(~np.isnan(R)) / (R.size)
#print('sparsity = {:.3f}'.format(sparsity))

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

# 计算这个矩阵每个用户的评分均值用于中心化
ru_mean = np.nanmean(R, axis=1)
#print('ru_mean = {}'.format(ru_mean))

# 将前面计算的用户评分均值变成向量，相当于转置，因为原来评分均值是行向量
ru_mean_vector = ru_mean.reshape((ru_mean.size, 1))
#print('ru_mean = \n{}'.format(ru_mean.reshape((ru_mean.size, 1))))

# 将评分矩阵进行中心化操作，方便后面的类似度计算
R_center = R - ru_mean_vector
#print('R_center\' = \n{}'.format(R_center))

# 定义调整余弦相似度计算函数
def discounted_pearson(u, v):
    """
    評価値行列R_centerにおけるアイテムiとアイテムjの調整コサイン類似度を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    j : int
        アイテムjのID

    Returns
    -------
    cosine : float
        調整コサイン類似度
    """
    Iuv = np.intersect1d(Iu[u], Iu[v])
    if Iuv.size <= 0: 
        return np.nan
    num = np.sum([R_center[u,i] * R_center[v,i] for i in Iuv])
    den_u = np.sqrt(np.sum([R_center[u,i]**2 for i in Iuv]))
    den_v = np.sqrt(np.sum([R_center[v,i]**2 for i in Iuv]))
    if den_u == 0 or den_v == 0: 
        return np.nan
    pearson = num / (den_u * den_v)
    dis_pearson = pearson * min(Iuv.size,β_u) / β_u
    if Iuv.size < β_u: 
        return dis_pearson
    return pearson

# 定义相似度函数，这里返回值为余弦相似度
def sim(u, v):
    """
    アイテム類似度関数:アイテムiとアイテムjのアイテム類似度を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    j : int
        アイテムjのID

    Returns
    -------
    float
        アイテム類似度
    """
    return discounted_pearson(u, v)

# 生成相似度矩阵函数
S = np.zeros((U.size, U.size))
for u in U:
    for v in range(u, U.size):
        if u == v:
            S[u,v] = 1.0
        else:
            # R[:, i] は アイテム i に関する全ユーザの評価を並べた列ベクトル
            S[u,v] = sim(u, v)
            S[v,u] = S[u,v]
#——————————————————————————————————————————————————————————————————
#pprint.pprint(S)
#print('S = \n{}'.format(S))
#——————————————————————————————————————————————————————————————————

# 详细生成每个物品对其他物品的类似度的词典
Uu = {u: {v: S[u,v] for v in U if u!=v} for u in U}
#print('Ii = ')
#pprint.pprint(Ii)

# 相似度最大的k个物品集合
Uu_k_user = {u: dict(sorted(Uu[u].items(), key=lambda x:x[1], reverse=True)[:K_USERS]) for u in U}
#print('Ii_k_item = ')
#pprint.pprint(Ii_k_item)

# 从k个物品集合里面剔除相似度不超过阈值，即相似度为负数的物品
Uu_k_user_plus_theta = {u: {v:s for v,s in Uu_k_user[u].items() if s > THETA} for u in U}
#print('Ii_k_item_plus_theta = ')
#pprint.pprint(Ii_k_item_plus_theta)

# 汇总对应物品的近邻物品的索引集合
Nu = {u: np.array(list(Uu_k_user_plus_theta[u].keys())) for u in U}
#print('Ni = ')
#pprint.pprint(Ni)

def predict(u, i):
    """
    予測関数:ユーザuのアイテムiに対する予測評価値を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    i : int
        アイテムiのID
    
    Returns
    -------
    float
        ユーザuのアイテムiに対する予測評価値
    """
    # 找到对应物品的近邻物品中u评价过的物品的索引集合
    Uui = np.intersect1d(Ui[i], Nu[u])
    # print('I{}{} = {}'.format(i, u, Iiu))
    if Uui.size <= 0: return ru_mean[u]
    # 评分预测值
    num = np.sum([(S[u,v] * R_center[v,i]) for v in Uui])
    den = np.sum([np.abs(S[u,v]) for v in Uui])
    rui_pred = ru_mean[u] + num / den
    
    return rui_pred

#读取测试集的数据，并生成对应的评分误差
df_u1_test = pd.read_csv('u1.test', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])
for i in df_u1_test.index:
   row = df_u1_test.loc[i]
   RMSE += math.pow( predict(row['user_id'] -1 , row['item_id'] - 1) - row['rating'] , 2 )
RMSE = math.sqrt(RMSE/len(df_u1_test.index))
print('RMSE = {}'.format(RMSE))
print('K_USERS = {}'.format(K_USERS))
print('THETA = {}'.format(THETA))
print('β_u = {}'.format(β_u))

# #生成一个只有预测评分的矩阵
# R_pred = np.nan * np.ones(R_center.shape) 
# for u in U:
#     for i in Iu_not[u]:
#         R_pred[u][i] = predict(u, i)
# print(R_pred)
# print(S)

