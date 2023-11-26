import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from class_similarity_matrix import SimilarityMatrixGenerator
from itembase_predict import R, Ui, β_u, R_topN1_predict, R_test, threshold, total_users, total_items
from itembase_mmr import lambda_constant, topN3


# 测试用的R矩阵
#——————————————————————————————————————————————————————————————————
# R = np.array([
#     [1, 2, 1, np.nan, 3, 3],
#     [3, np.nan, 4, 5, np.nan, 5],
#     [1, 4, 5, np.nan, 3, 2],
#     [1, np.nan, 4, 2, 5, np.nan],
# ])
#——————————————————————————————————————————————————————————————————

# 求出物品之间的相似度，这里表示为矩阵的形式
generator = SimilarityMatrixGenerator(R, β_u)
S = generator.generate_similarity_matrix()
# sparsity = 1 - np.count_nonzero(~np.isnan(S)) / (S.size)

# 这里定义了一个目标函数计算准确性和多样性的trade-off
def objective_function(current_recommendations, lambda_constant, similarity_matrix):
    R_size = len(current_recommendations)
    
    # Calculate the contribution of user ratings (r(u, i)) to the objective function.
    #——————————————————————————————————————————————————————————————————2023.7.22 进度，这里把评分数据标准化了。（又删除了）
    rating_contribution = lambda_constant * sum(score for _, score in current_recommendations) / R_size 
    
    # Calculate the contribution of item similarities (sim(i, j)) to the objective function.
    similarity_contribution = 0.0

    if R_size > 1:
        for i, _ in current_recommendations:
            for j, _ in current_recommendations:
                if i != j:
                    similarity_contribution += similarity_matrix[i][j]

        similarity_contribution *= (1 - lambda_constant) / (R_size * (R_size - 1))
    
    # Calculate the overall objective function value.
    obj_value = rating_contribution - similarity_contribution
    
    return obj_value

# 贪心算法计算准确性和多样性的trade-off
def greedy_algorithm(R_topN1_predict, lambda_constant, topN3, similarity_matrix):
    final_recommendations = {}
    for user, item_scores in R_topN1_predict.items():
        R = []
        while len(R) < topN3:
            best_item = None
            best_objective = float('-inf')
            
            for item, score in item_scores.items():
                if item not in [rec[0] for rec in R]:
                    # Calculate the objective function value if we add this item to R.
                    current_recommendations = R + [(item, score)]
                    current_objective = objective_function(current_recommendations, lambda_constant, similarity_matrix)
                    
                    if current_objective > best_objective:
                        best_item = (item, score)
                        best_objective = current_objective
            
            if best_item:
                R.append(best_item)
            else:
                # If no improvement can be made, stop the greedy optimization.
                break
        
        final_recommendations[user] = [item for item, _ in R]
    
    return final_recommendations

final_recommendations = greedy_algorithm(R_topN1_predict, lambda_constant, topN3, S)



# 计算推荐结果的覆盖率
def calculate_coverage(final_recommendations, total_items):
    recommended_items = set()

    for user_recommendations in final_recommendations.values():
        recommended_items.update(user_recommendations)

    coverage = len(recommended_items) / total_items
    return coverage

coverage = calculate_coverage(final_recommendations, total_items)

print('coverage = {}'.format(coverage))

# 计算推荐结果的新奇性
def calculate_novelty(recommendations, Ui, total_users):
    novelty_scores = []

    for user_recommendations in recommendations.values():
        novelty_score = 0.0

        for item in user_recommendations:

            Ui_item_length = len(Ui[item])

            novelty_score += 1 - (Ui_item_length / total_users)

        novelty_score /= len(user_recommendations)
        novelty_scores.append(novelty_score)

    average_novelty = sum(novelty_scores) / len(novelty_scores)
    return average_novelty

average_novelty = calculate_novelty(final_recommendations, Ui, total_users)

print('average_novelty = {}'.format(average_novelty))

# 计算推荐结果的多样性
def compute_diversity(final_recommendations, S):
    total_diversity_score = 0

    for user, recommended_items in final_recommendations.items():
        diversity_score = 0
        n = len(recommended_items)
        total_pairs = n * (n - 1) 

        for i in recommended_items:
            for j in recommended_items:
                if i != j:
                    similarity = S[i][j]
                    diversity_score += 1 - similarity

        diversity_score /= total_pairs
        total_diversity_score += diversity_score

    average_diversity_score = total_diversity_score / total_users
    return average_diversity_score

average_diversity = compute_diversity(final_recommendations, S)

print('average_diversity = {}'.format(average_diversity))

# 计算推荐结果的准确性

# 首先统计测试数据集中用户评价过的物品集合和测试数据集中用户评价过的超过阈值的物品集合
def create_test_ratings_and_over_threshold_from_matrix(R_test, threshold):

    test_ratings = {}
    test_ratings_over_threshold = {}

    for user_idx, user_ratings in enumerate(R_test):

        rated_items = [item_idx for item_idx, rating in enumerate(user_ratings) if not math.isnan(rating)]
        high_rated_items = [item_idx for item_idx, rating in enumerate(user_ratings) if rating >= threshold]

        test_ratings[user_idx] = rated_items
        test_ratings_over_threshold[user_idx] = high_rated_items
        
    return test_ratings, test_ratings_over_threshold

# 计算precision
def calculate_precision(final_recommendations, R_test):

    test_ratings, test_ratings_over_threshold = create_test_ratings_and_over_threshold_from_matrix(R_test, threshold)

    total_items_count = 0
    total_correct = 0

    for user, recommended_items in final_recommendations.items():

        actual_items_over_threshold = test_ratings_over_threshold[user]
        actual_items = test_ratings[user]
        
        correct_items = [item for item in recommended_items if item in actual_items_over_threshold]
        total_items = [item for item in recommended_items if item in actual_items]

        total_correct += len(correct_items)
        total_items_count += len(total_items)

    precision = total_correct / total_items_count
    
    return precision

precision = calculate_precision(final_recommendations, R_test)



































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
