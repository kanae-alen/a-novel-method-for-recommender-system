import numpy as np
import pandas as pd
from numpy.linalg import norm

#读取数据，并且生成一个与数据维度相同的0矩阵
df = pd.read_csv('u.data', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])
shape = (df.max().loc['user_id'], df.max().loc['item_id'])
R = np.zeros(shape) 

#将数据中的元素一个一个赋值给0矩阵，形成完整的评分矩阵
for i in df.index:
    row = df.loc[i]
    R[row['user_id'] -1 , row['item_id'] - 1] = row['rating']

#这个E矩阵是测试用
E = np.array([[1, 2, 1, 0, 3, 3], 
    [3, 0, 4, 5, 0, 5],
    [1, 4, 5, 0, 3, 2],
    [1, 0, 4, 2, 5, 0]])

#这一步是生成物品与物品之间的相似度矩阵
def compute_item_similarities(R):
    # n: movie counts
    n = R.shape[1]
    sims = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                # R[:, i] は アイテム i に関する全ユーザの評価を並べた列ベクトル
                sim = similarity(R[:,i], R[:,j])
            sims[i][j] = sim 
            sims[j][i] = sim 
    return sims 

#这一步是物品评分减去用户评分倾向造成的偏差(这里后面肯定还需要优化，因为这里的用户评分平均值计算都是在线计算，后面肯定要转为离线计算)
def user_meanrate(user_index, R):
    user_meanrating = []
    for i in user_index:
        user_meanrating.append(sum(R[i,:])/len((np.where(R[i,:] != 0))[0]))
    return user_meanrating

#这一步是生成物品与物品之间的相似度矩阵函数的子函数，是具体如何计算相似度的核心流程（后面应该需要在这里加上各种制约条件和阈值函数）
def similarity(item1, item2):
    # item1 と item2 のどちらも評価済であるユーザの集合
    common = np.logical_and(item1 != 0, item2 != 0)
    user_index = np.where(common == 1)
    user_index = user_index[0]
    user_meanrating = user_meanrate(user_index, E)
    v1 = item1[common] - user_meanrating
    v2 = item2[common] - user_meanrating
    sim = 0.0
    # 共通評価者が 2以上という制約にしている
    if v1.size > 1:
        sim = np.dot(v1,v2)/(norm(v1)*norm(v2))
        

    return sim

sims = compute_item_similarities(E)
print(sims)


