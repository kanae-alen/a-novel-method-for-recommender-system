import numpy as np
np.set_printoptions(precision=3)

# 测试用的R矩阵
R = np.array([
              [1, 2,      1,      np.nan,      3,      3],
              [3,      np.nan,      4,      5, np.nan,      5     ],
              [1,      4, 5,     np.nan,      3,      2],
              [1, np.nan,      4, 2,      5,      np.nan     ],
])

# 评价两个物品的共同用户的数目阈值
β_u = 40

# 生成用户索引集合
U = np.arange(R.shape[0])
#print('U = {}'.format(U))

# 生成物品索引集合
I = np.arange(R.shape[1])
#print('I = {}'.format(I))

# 生成评价过对应物品的用户索引集合
Ui = [U[~np.isnan(R)[:,i]] for i in I]
#print('Ui = ')
#pprint.pprint(Ui)

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
def discounted_adjusted_cos(i, j):
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
    Uij = np.intersect1d(Ui[i], Ui[j])
    if Uij.size <= 0: 
        return np.nan
    num = np.sum([R_center[u,i] * R_center[u,j] for u in Uij])
    den_i = np.sqrt(np.sum([R_center[u,i]**2 for u in Uij]))
    den_j = np.sqrt(np.sum([R_center[u,j]**2 for u in Uij]))
    if den_i == 0 or den_j == 0: 
        return np.nan
    cosine = num / (den_i * den_j)
    cosine_β_u = cosine * min(Uij.size,β_u) / β_u
    if Uij.size < β_u: 
        return cosine_β_u
    return cosine

# 定义相似度函数，这里返回值为余弦相似度
def sim(i, j):
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
    return discounted_adjusted_cos(i, j)

# 生成相似度矩阵函数
S = np.zeros((I.size, I.size))
for i in I:
    for j in range(i, I.size):
        if i == j:
            S[i,j] = 1.0
        else:
            # R[:, i] は アイテム i に関する全ユーザの評価を並べた列ベクトル
            S[i,j] = sim(i, j)
            S[j,i] = S[i,j]

print('S = {}'.format(S))