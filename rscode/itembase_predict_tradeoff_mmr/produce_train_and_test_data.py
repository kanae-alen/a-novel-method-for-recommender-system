import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取数据集
data = pd.read_csv('D:/rscode/tradeoff_mmr/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# 2. 去除时间戳列
data = data.drop(columns=['timestamp'])

# 3. 按用户分组
grouped_data = data.groupby('user_id')

# 4. 划分数据
train_data = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
test_data = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

for user_id, group in grouped_data:
    # 获取用户评分数据
    user_ratings = group[['user_id', 'item_id', 'rating']]
    
    # 划分数据，设置测试集占比为20%
    train_ratings, test_ratings = train_test_split(user_ratings, test_size=0.2, random_state=42)
    
    # 合并到新的数据集中
    train_data = pd.concat([train_data, train_ratings])
    test_data = pd.concat([test_data, test_ratings])

# 5. 打印数据集的大小（可选，查看训练集和测试集大小）
print("Train data size:", len(train_data))
print("Test data size:", len(test_data))

# 6. 保存新数据集
train_data.to_csv('train_data.txt', sep='\t', index=False, header=None, mode='a')
test_data.to_csv('test_data.txt', sep='\t', index=False, header=None, mode='a')





































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
