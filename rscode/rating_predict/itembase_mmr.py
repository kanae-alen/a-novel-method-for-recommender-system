import numpy as np
from similarity_matrix import S

R_predict = np.array([
              [1, 2,      1,      np.nan,      3,      3],
              [3,      np.nan,      4,      5, np.nan,      5     ],
              [1,      4, 5,     np.nan,      3,      2],
              [1, np.nan,      4, 2,      5,      np.nan     ],
])

ratings_dict = {}

for i in range(R.shape[0]):
    user_ratings = {}
    for j in range(R.shape[1]):
        if not np.isnan(R[i, j]):
            user_ratings[f"物品{j}"] = R[i, j]
    ratings_dict[f"用户{i}"] = user_ratings

print('ratings_dict = {}'.format(ratings_dict))