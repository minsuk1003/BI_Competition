import pandas as pd
import os
import numpy as np
import networkx as nx
from collections import Counter
from util import error_masking, usage_masking

def df_processing(data_list, data_path):
    total_df = pd.DataFrame()
    
    for i in data_list:
        df = pd.read_csv(data_path + f'/{i}')
        df = df.astype({'Total_Money':'int64', 'Total_Best_Money':'int64', 'Money_percent' : 'float64'})
        index_max = (max(df.index) + 1)
        K_number = len(df)
        for j in range(len(df)):
            total, medium, small = df.Total_volume[j].split('(')[1].split(')')[0].split(',')
            df.loc[j, 'total_volume'] = float(total)
            df.loc[j, 'medium_volume'] = float(medium)
            df.loc[j, 'small_volume'] = float(small)

        df.drop('Total_volume', axis=1, inplace=True)
        df.loc[index_max, 'item_complex'] = (f'Complex_Name = {i}')
        df.loc[index_max, 'Total_Money'] = df.Total_Money.sum()
        df.loc[index_max, 'Total_Best_Money'] = df.Total_Best_Money.sum()
        df.loc[index_max, 'Money_percent'] = (df.Total_Money.sum() / df.Total_Best_Money.sum())

        df.loc[index_max, 'total_volume'] = df.total_volume.sum()
        df.loc[index_max, 'medium_volume'] = df.medium_volume.sum()
        df.loc[index_max, 'small_volume'] = df.small_volume.sum()
        df.loc[index_max, 'Number_K'] = K_number
        
        total_df = pd.concat([total_df, df], ignore_index=True)

    return total_df


def graph_add_tuple_weights(df, col_name='item_complex'):
    tuples = []
    for i in df[col_name]:
        i = i.split('[')[1].split(']')[0].replace("'", '').replace(' ', '').split(',')
        lst = list(i)
        
        # 튜플 쌍 생성
        temp_tuples = []
        temp_tuples = [(lst[m], lst[n]) for m in range(len(lst)) for n in range(m+1, len(lst))]
        tuples = tuples + temp_tuples

    # 튜플 빈도수 계산
    tuple_counts = Counter(tuples)

    # 그래프 생성
    G = nx.Graph()

    # 중복되는 튜플인 경우 가중치를 증가시키면서 엣지 추가
    for tuple_pair, count in tuple_counts.items():
        G.add_edge(tuple_pair[0], tuple_pair[1], weight=count)

    return G

def sum_make(index_num, item_list, df):
    temp_df = pd.DataFrame()
    for i in item_list:
        dataframe = df.copy()
        dataframe = dataframe[dataframe['Power_Code'] == i]
        dataframe.Date = pd.to_datetime(dataframe.Date)
        dataframe = dataframe.sort_values(by = ['Date'])
        dataframe.reset_index(inplace=True, drop=True)
        temp_df = pd.concat([temp_df, dataframe.loc[index_num:index_num, :]])

    volume_sum = temp_df.iloc[:, 3].sum()
    real_sum = temp_df.iloc[:, 4:28].sum()
    pred_sum = temp_df.iloc[:, 29:53].sum()

    use_masking = usage_masking(np.array(real_sum) / volume_sum)
    error_rate = (abs(np.array(pred_sum) - np.array(real_sum)) / volume_sum)
    err_masking = error_masking(abs(np.array(pred_sum) - np.array(real_sum)) / volume_sum)

    return np.array(real_sum), np.array(pred_sum), volume_sum, error_rate