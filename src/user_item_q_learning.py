import numpy as np
import pandas as pd
import random

def create_ratings_dataframe():
    """创建用户-物品评分矩阵"""
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
        'item_id': [1, 2, 3, 1, 3, 2, 3, 4],
        'rating': [5, 3, 2, 4, 5, 1, 4, 5]
    }
    return pd.DataFrame(ratings_data)

def initialize_q_table(users, items):
    """初始化Q表"""
    return pd.DataFrame(np.zeros((len(users), len(items))), index=users, columns=items)

def q_learning(ratings_df, q_table, learning_rate, discount_factor, exploration_rate, epochs):
    """执行Q-learning算法"""
    users = ratings_df['user_id'].unique()
    items = ratings_df['item_id'].unique()

    for _ in range(epochs):
        for user in users:
            if random.uniform(0, 1) < exploration_rate:
                item = random.choice(items)  # 探索
            else:
                item = q_table.loc[user].idxmax()  # 利用

            rating = get_rating(ratings_df, user, item)
            update_q_value(q_table, user, item, rating, learning_rate, discount_factor)

    return q_table

def get_rating(ratings_df, user, item):
    """获取用户对物品的评分"""
    rating = ratings_df[(ratings_df['user_id'] == user) & (ratings_df['item_id'] == item)]['rating'].values
    return rating[0] if len(rating) > 0 else 0

def update_q_value(q_table, user, item, rating, learning_rate, discount_factor):
    """更新Q值"""
    current_q = q_table.loc[user, item]
    max_future_q = q_table.loc[user].max()
    new_q = current_q + learning_rate * (rating + discount_factor * max_future_q - current_q)
    q_table.loc[user, item] = new_q

def recommend_item(user_id, q_table):
    """为用户推荐Q值最高的物品"""
    return q_table.loc[user_id].idxmax()

def main():
    # 创建评分数据
    ratings_df = create_ratings_dataframe()
    print("评分数据:")
    print(ratings_df)

    # 获取用户和物品列表
    users = ratings_df['user_id'].unique()
    items = ratings_df['item_id'].unique()

    # 初始化Q表
    q_table = initialize_q_table(users, items)

    # 定义参数
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 0.2
    epochs = 1000

    # 执行Q-learning算法
    q_table = q_learning(ratings_df, q_table, learning_rate, discount_factor, exploration_rate, epochs)

    print('\nQ表:')
    print(q_table)

    # 为每个用户推荐物品
    for user in users:
        recommended_item = recommend_item(user, q_table)
        print(f"为用户 {user} 推荐物品 {recommended_item}")

    # 用户反馈
    user_feedback = {
        1: {1: 1, 2: 0, 3: 0},
        2: {1: 0, 2: 1, 3: 1},
        3: {1: 1, 2: 1, 3: 0}
    }

    print("\n用户反馈:")
    for user, feedback in user_feedback.items():
        print(f"用户 {user} 的反馈: {feedback}")

if __name__ == "__main__":
    main()