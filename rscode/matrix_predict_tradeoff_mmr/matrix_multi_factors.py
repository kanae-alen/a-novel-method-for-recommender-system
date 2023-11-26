import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from class_similarity_matrix import SimilarityMatrixGenerator
from matrix_predict import R, Ui, β_u, R_topN1_predict, R_test, threshold, total_users, total_items

from matrix_mmr import lambda_constant, topN2
from matrix_mmr import lambda_tradeoff
from matrix_mmr import topN3

from matrix_mmr import precision as precision_mmr
from matrix_mmr import average_diversity as average_diversity_mmr
from matrix_mmr import average_novelty as average_novelty_mmr
from matrix_mmr import coverage as coverage_mmr

from matrix_accuracy_diversity_tradeoff import precision as precision_acc_div_tradeoff
from matrix_accuracy_diversity_tradeoff import average_diversity as average_diversity_acc_div_tradeoff
from matrix_accuracy_diversity_tradeoff import average_novelty as average_novelty_acc_div_tradeoff
from matrix_accuracy_diversity_tradeoff import coverage as coverage_acc_div_tradeoff
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

# 将筛选后的topN2的物品根据他们的评分进行排序，分数越大的越排在前面
def sort_recommendations_by_ratings(recommendations):
    sorted_recommendations = {}
    for user, item_scores in recommendations.items():
        # 对物品按评分从高到低进行排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        # 记录排名的变量
        ranked_items = {}
        rank = 0
        prev_score = None

        for i, (item, score) in enumerate(sorted_items):
            if score != prev_score:
                # 分数和上一个物品不同，更新排名
                prev_score = score
                rank += 1
            # 分数相同的物品赋予相同排名
            ranked_items[item] = rank

        sorted_recommendations[user] = ranked_items

    return sorted_recommendations

recommendation_step2_rating_sorted = sort_recommendations_by_ratings(R_topN1_predict)

# 首先生成一个评价过物品的用户数替换掉评分数据的字典
def replace_ratings_with_novelty(original_data, Ui):

    new_data = {}

    # 遍历原始数据中的每个用户和其对应的评分数据
    for user_id, item_ratings in original_data.items():
        # 获取用户已评价过的物品列表
        rated_items = item_ratings.keys()

        # 从物品评价用户人数的字典中获取这些物品的评价用户人数
        new_ratings = {item_id: len(Ui[item_id]) for item_id in rated_items}

        # 将替换后的评价用户人数添加到新数据字典中
        new_data[user_id] = new_ratings

    return new_data

recommendation_replace_ratings_with_novelty = replace_ratings_with_novelty(recommendation_step2_rating_sorted, Ui)

# 将筛选后的topN2的物品根据他们的新奇度进行排序，评分过该物品的用户数越少，该物品越排在前面
def sort_recommendations_by_novelty(recommendations):
    sorted_recommendations = {}
    for user, item_scores in recommendations.items():
        
        sorted_item_scores = item_scores.copy()
        # 对物品按评分从高到低进行排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1])
        
        # 记录排名的变量
        
        rank = 0
        prev_score = None

        for i, (item, score) in enumerate(sorted_items):
            if score != prev_score:
                # 分数和上一个物品不同，更新排名
                prev_score = score
                rank += 1
            # 分数相同的物品赋予相同排名
            sorted_item_scores[item] = rank

        sorted_recommendations[user] = sorted_item_scores
    return sorted_recommendations

recommendation_step2_novelty_sorted = sort_recommendations_by_novelty(recommendation_replace_ratings_with_novelty)

# 将两个排序结果进行trade-off得到一个新的排序
def tradeoff_rankings(rating_rankings, novelty_rankings, lambda_tradeoff):
    tradeoff_rankings = {}

    for user in rating_rankings:
        tradeoff_rankings[user] = {}

        for item in rating_rankings[user]:
            # Combine rankings using trade-off
            tradeoff_score = lambda_tradeoff * rating_rankings[user][item] + (1 - lambda_tradeoff) * novelty_rankings[user][item]
            tradeoff_rankings[user][item] = tradeoff_score

    return tradeoff_rankings

recommendation_step3_tradeoff = tradeoff_rankings(recommendation_step2_rating_sorted, recommendation_step2_novelty_sorted, lambda_tradeoff)

# 生成最终的推荐结果
def generate_final_recommendations(tradeoff_rankings, topN3):
    final_recommendations = {}

    for user, item_rankings in tradeoff_rankings.items():
        sorted_items = sorted(item_rankings.items(), key=lambda x: x[1])
        top_items = [item for item, _ in sorted_items[:topN3]]
        final_recommendations[user] = top_items

    return final_recommendations

final_recommendations = generate_final_recommendations(recommendation_step3_tradeoff, topN3)

# 计算推荐结果的覆盖率
def calculate_coverage(final_recommendations, total_items):
    recommended_items = set()

    for user_recommendations in final_recommendations.values():
        recommended_items.update(user_recommendations)

    coverage = len(recommended_items) / total_items
    return coverage

coverage_multi_factors = calculate_coverage(final_recommendations, total_items)

print('coverage_multi_factors = {}'.format(coverage_multi_factors))
print('coverage_mmr = {}'.format(coverage_mmr))
print('coverage_acc_div_tradeoff = {}'.format(coverage_acc_div_tradeoff))

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

average_novelty_multi_factors = calculate_novelty(final_recommendations, Ui, total_users)

print('average_novelty_multi_factors = {}'.format(average_novelty_multi_factors))
print('average_novelty_mmr= {}'.format(average_novelty_mmr))
print('average_novelty_acc_div_tradeoff= {}'.format(average_novelty_acc_div_tradeoff))

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

average_diversity_multi_factors = compute_diversity(final_recommendations, S)

print('average_diversity_multi_factors = {}'.format(average_diversity_multi_factors))
print('average_diversity_mmr = {}'.format(average_diversity_mmr))
print('average_diversity_acc_div_tradeoff = {}'.format(average_diversity_acc_div_tradeoff))

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

precision_multi_factors = calculate_precision(final_recommendations, R_test)

print('precision_multi_factors = {}'.format(precision_multi_factors))
print('precision_mmr = {}'.format(precision_mmr))
print('precision_acc_div_tradeoff = {}'.format(precision_acc_div_tradeoff))



































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
