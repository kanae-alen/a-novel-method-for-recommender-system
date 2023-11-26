import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from class_similarity_matrix import SimilarityMatrixGenerator
from itembase_predict import R, Ui, β_u, R_topN1_predict, R_test, threshold, total_users, total_items

# 测试用的R矩阵
#——————————————————————————————————————————————————————————————————
# R = np.array([
#     [1, 2, 1, np.nan, 3, 3],
#     [3, np.nan, 4, 5, np.nan, 5],
#     [1, 4, 5, np.nan, 3, 2],
#     [1, np.nan, 4, 2, 5, np.nan],
# ])
#——————————————————————————————————————————————————————————————————

# 准确性和多样性的trade-off系数(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
lambda_constant = 0.75
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 准确性和新奇性的trade-off系数(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
lambda_tradeoff = 0.50
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 选出准确性和多样性trade-off效果最好的topN2个物品(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#—————2023.7.22实验效果，topN2=10时mmr比multi-factor效果要好，考虑后续把topN2固定在一个比较小的数值
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
topN2 = 10
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 最终推荐生成topN3个物品(ps:这个很重要，后面感觉要不断调节这个参数来达到好的效果)
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!
topN3 = 5
#——————————————————————————————————————————————————————————————————!!!!!!!!!!!!!!!!

# 求出物品之间的相似度，这里表示为矩阵的形式
generator = SimilarityMatrixGenerator(R, β_u)
S = generator.generate_similarity_matrix()
# sparsity = 1 - np.count_nonzero(~np.isnan(S)) / (S.size)

# 做一个准确性和多样性的trade-off，从topN1个物品里面选出topN2个物品
# 关于topN1: 转换了策略，现在topN1不定，因为预测评分也是选择大于阈值threshold的评分
#——————————————————————————————————————————————————————————————————2023.7.20 进度，这里trade-off需要重新来写。

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
def greedy_algorithm(R_topN1_predict, lambda_constant, topN2, similarity_matrix):
    final_recommendations = {}
    for user, item_scores in R_topN1_predict.items():
        R = []
        while len(R) < topN2:
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
        
        final_recommendations[user] = {item: score for item, score in R}
    
    return final_recommendations

recommendation_step1_tradeoff = greedy_algorithm(R_topN1_predict, lambda_constant, topN2, S)

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

recommendation_step2_rating_sorted = sort_recommendations_by_ratings(recommendation_step1_tradeoff)

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

print('precision = {}'.format(precision))
