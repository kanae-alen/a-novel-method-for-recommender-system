import numpy as np
np.set_printoptions(precision=3)

class SimilarityMatrixGenerator:
    
    def __init__(self, R, β_u):
        self.R = R
        self.β_u = β_u
        self.U = np.arange(R.shape[0])
        self.I = np.arange(R.shape[1])
        self.Ui = [self.U[~np.isnan(R)[:,i]] for i in self.I]
        self.ru_mean = np.nanmean(R, axis=1)
        self.ru_mean_vector = self.ru_mean.reshape((self.ru_mean.size, 1))
        self.R_center = R - self.ru_mean_vector
        self.S = np.zeros((self.I.size, self.I.size))

    def discounted_adjusted_cos(self, i, j):
        Uij = np.intersect1d(self.Ui[i], self.Ui[j])
        if Uij.size <= 0: 
            return 0
        #——————————————2023.7.20 进度，重要修改，按照安元的方法将相似度矩阵的缺失值补为0。
        num = np.sum([self.R_center[u,i] * self.R_center[u,j] for u in Uij])
        den_i = np.sqrt(np.sum([self.R_center[u,i]**2 for u in Uij]))
        den_j = np.sqrt(np.sum([self.R_center[u,j]**2 for u in Uij]))
        if den_i == 0 or den_j == 0: 
            return 0
        #——————————————2023.7.20 进度，重要修改，按照安元的方法将相似度矩阵的缺失值补为0。
        cosine_β_u = ( num / (den_i * den_j) ) * ( min(Uij.size, self.β_u) / self.β_u )
        return cosine_β_u

    def sim(self, i, j):
        return self.discounted_adjusted_cos(i, j)

    def generate_similarity_matrix(self):
        for i in self.I:
            for j in range(i, self.I.size):
                if i == j:
                    self.S[i, j] = 1.0
                else:
                    self.S[i, j] = self.sim(i, j)
                    self.S[j, i] = self.S[i, j]

        return self.S