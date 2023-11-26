import numpy as np
import math
import pandas as pd
import scipy.stats as stats
# topN = 5


df_u1_train = pd.read_csv('D:/rscode/tradeoff_mmr/train_data.txt', sep='\t', names=['user_id','item_id', 'rating'])
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

print(R)




































# def sort_recommendations_by_novelty(recommendations, Ui, total_users):
#     sorted_recommendations = {}
    
#     for user, sorted_items in recommendations.items():
#         novelty_score = len(Ui[i]) / total_users
#         sorted_recommendations[user] = (sorted_items, novelty_score)
    
#     sorted_recommendations = dict(sorted(sorted_recommendations.items(), key=lambda x: x[1][1]))
    
#     return sorted_recommendations


# recommendation_step3_novelty_sorted = sort_recommendations_by_novelty(recommendation_step2_sorted, Ui, total_users)






# user, predictions = R_topN_predict.items()
# items = list(predictions.keys())


# print('user = {}'.format(user))
# print('predictions = {}'.format(predictions))
# print('items = {}'.format(items))
# for user, user_ratings in R_topN_predict.items():
#         selected_items = []
#         remaining_items = list(user_ratings.keys())
#         print('remaining_items = {}'.format(remaining_items))
#         print(user)


# def greedy_tradeoff_recommendation(R_topN1_predict, S, lambda_constant=0.5, topN2=3):
#     recommendations = {}
    
#     for user, user_ratings in R_topN1_predict.items():
#         selected_items = []
#         remaining_items = list(user_ratings.keys())
        
#         while len(selected_items) < topN2 and remaining_items:
#             best_score = float("-inf")
#             best_item = None
            
#             for item in remaining_items:
#                 item_score = user_ratings[item]

#                 similarity_score = 0

#                 for selected_item in selected_items:
#                     if not math.isnan(S[selected_item][item]):
#                         similarity_score += S[selected_item][item]
                        
#                 average_item_score = item_score / topN2
#                 average_similarity_score = 2 * similarity_score / (topN2 * (topN2 - 1))
                
#                 tradeoff_score = lambda_constant * average_item_score - (1 - lambda_constant) * average_similarity_score
                
#                 if tradeoff_score > best_score:
#                     best_score = tradeoff_score
#                     best_item = item
            
#             remaining_items.remove(best_item)
#             selected_items.append(best_item)
        
#         recommendations[user] = selected_items
    
#     return recommendations





# def greedy_tradeoff_recommendation(ratings, similarities, lambda_constant=0.5, k=3):
#     recommendations = {}
    
#     for user, user_ratings in ratings.items():
#         selected_items = []
#         remaining_items = list(user_ratings.keys())
        
#         while len(selected_items) < k and remaining_items:
#             best_score = float("-inf")
#             best_item = None
#             similarity_score = 0
#             count = 0
            
#             for item in remaining_items:
#                 item_score = user_ratings[item]
                
#                 for selected_item in selected_items:
#                     if selected_item in similarities and item in similarities[selected_item]:
#                         similarity_score += similarities[selected_item][item]
                        
#                 average_item_score = item_score / k
#                 average_similarity_score = 2 * similarity_score / ( k * ( k - 1 ) ) 
                
#                 tradeoff_score = lambda_constant * average_item_score - (1 - lambda_constant) * average_similarity_score
                
#                 if tradeoff_score > best_score:
#                     best_score = tradeoff_score
#                     best_item = item
            
#             remaining_items.remove(best_item)
#             selected_items.append(best_item)
        
#         recommendations[user] = selected_items
    
#     return recommendations



# def greedy_tradeoff_recommendation(ratings, similarities, lambda_constant=0.5, k=3):
#     recommendations = {}
    
#     for user, user_ratings in ratings.items():
#         selected_items = []
#         remaining_items = list(user_ratings.keys())
        
#         while len(selected_items) < k and remaining_items:
#             best_score = float("-inf")
#             best_item = None
            
#             for item in remaining_items:
#                 item_score = user_ratings[item]

#                 similarity_score = 0

#                 for selected_item in selected_items:
#                     if selected_item in similarities and item in similarities[selected_item]:
#                         similarity_score += similarities[selected_item][item]
                        
#                 average_item_score = item_score / k
#                 average_similarity_score = 2 * similarity_score / ( k * ( k - 1 ) )  
                
#                 tradeoff_score = lambda_constant * average_item_score - (1 - lambda_constant) * average_similarity_score
                
#                 if tradeoff_score > best_score:
#                     best_score = tradeoff_score
#                     best_item = item
            
#             remaining_items.remove(best_item)
#             selected_items.append(best_item)
        
#         recommendations[user] = selected_items
    
#     return recommendations









# def greedy_tradeoff_recommendation(ratings, similarities, lambda_constant=0.5, k=3):
#     recommendations = {}
    
#     for user, user_ratings in ratings.items():
#         selected_items = []
#         remaining_items = list(user_ratings.keys())
        
#         while len(selected_items) < k and remaining_items:
#             best_score = float("-inf")
#             best_item = None
            
#             for item in remaining_items:
#                 item_score = user_ratings[item]
                
#                 similarity_score = 0
#                 for selected_item in selected_items:
#                     similarity_score = max(similarity_score, similarities[item][selected_item])
                
#                 tradeoff_score = lambda_constant * item_score - (1 - lambda_constant) * similarity_score
                
#                 if tradeoff_score > best_score:
#                     best_score = tradeoff_score
#                     best_item = item
            
#             remaining_items.remove(best_item)
#             selected_items.append(best_item)
        
#         recommendations[user] = selected_items
    
#     return recommendations
