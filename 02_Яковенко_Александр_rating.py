# Скрипт повторяет ячейку из ноутбука, подсчитывает рейтинги каждого игрока в каждый из "дней"
# и складывает данные в json-файлы для дальнейшей обработки

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
import scipy.sparse
import json

train_data = pd.read_csv('train.csv').set_index('id')
test_data = pd.read_csv('test.csv').set_index('id')

col_names = ['game_mode', 'id_1', 'rating_1', 'id_2', 'rating_2']
col_names += [f"unit{u_n}_{p_n}" for p_n in range(1,3) for u_n in range(1,9)]
col_names += ['game_duration']

train_data.rename(columns={k: v for k, v in zip(train_data.columns, col_names)}, inplace=True)
test_data.rename(columns={k: v for k, v in zip(test_data.columns, col_names)}, inplace=True)

concated_data = pd.concat((train_data, test_data))
sorted_concated_data = concated_data.sort_values(by=['game_duration', 'id'])

# calculate rating statistics for all days for each player
rating_stats = {}

for idx, player_id in tqdm.tqdm(enumerate(set(sorted_concated_data.id_1) | set(sorted_concated_data.id_2))):
    if idx // 100000 < 9:
        continue
    if idx % 100000 == 0:
        continue
        with open(f'_ratings_{idx // 100000}.json', 'w') as f:
            json.dump(rating_stats, f)
            rating_stats = {}
    
    df_ = sorted_concated_data[(sorted_concated_data.id_1 == player_id) | (sorted_concated_data.id_2 == player_id)]
    df_1 = df_[df_.id_1 == player_id]
    df_2 = df_[df_.id_2 == player_id]
    days = df_.game_duration.unique()
    
    rt1 = df_1.rating_1.tolist()
    rt2 = df_2.rating_2.tolist()
    ratings = rt1 + rt2
    
    if not rt1:
        rt1 = [-1]
    if not rt2:
        rt2 = [-1]
    if not ratings:
        ratings = [-1]
    
    app = {
        "mean_rt_all": float(np.mean(ratings)),
        "std_rt_all": float(np.std(ratings)),
#         "mean_rt_all_1": float(np.mean(rt1)),
#         "std_rt_all_1": float(np.std(rt1)),
#         "mean_rt_all_2": float(np.mean(rt2)),
#         "std_rt_all_2": float(np.std(rt2))
    }
#     break
    for day in days:
        df_1_d = df_1[df_1.game_duration == day]
        df_2_d = df_2[df_2.game_duration == day]
        rt1 = df_1_d.rating_1.tolist()
        rt2 = df_2_d.rating_2.tolist()
        ratings = rt1 + rt2
        
        if not rt1:
            rt1 = [-1]
        if not rt2:
            rt2 = [-1]
        if not ratings:
            ratings = [-1]
        
        app.update({
            int(day): {
                "mean": float(np.mean(ratings)),
                "std": float(np.std(ratings)),
                "min": float(np.min(ratings)),
                "max": float(np.max(ratings)),
#                 "mean_1": float(np.mean(rt1)),
#                 "std_1": float(np.std(rt1)),
#                 "min_1": float(np.min(rt1)),
#                 "max_1": float(np.max(rt1)),
#                 "mean_2": float(np.mean(rt2)),
#                 "std_2": float(np.std(rt2)),
#                 "min_2": float(np.min(rt2)),
#                 "max_2": float(np.max(rt2))
            }
        })
        
    rating_stats[int(player_id)] = app
#     break

with open(f'ratings_10.json', 'w') as f:
    json.dump(rating_stats, f)