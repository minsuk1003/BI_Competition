import pandas as pd
import numpy as np
import random

def set_item(Power_Code, Dataframe):
    dataframe = Dataframe.copy()
    dataframe = dataframe[dataframe['Power_Code'] == Power_Code]
    dataframe.Date = pd.to_datetime(dataframe.Date)
    dataframe = dataframe.sort_values(by = ['Date'])
    dataframe.reset_index(inplace=True, drop=True)

    power_volume = (dataframe.iloc[0, 3])
    real_array = np.array(dataframe.iloc[:, 4:28])
    pred_array = np.array(dataframe.iloc[:, 29:53])
    return power_volume, real_array, pred_array

# 사용량이 0.1보다 작은지 검증
def usage_masking(array):
    mask = array < 0.1
    array[mask] = 0
    array[~mask] = 1
    return array

def error_cal(volume, real, pred):
    value = (abs(real-pred)) / (volume)
    return value

def error_masking(array):
    mask1 = array > 0.08
    mask2 = np.logical_and(array > 0.06, array <= 0.08)
    mask3 = array <= 0.06
    array[mask1] = 0
    array[mask2] = 3
    array[mask3] = 4
    return array

class Processing:
    def __init__(self, Power_Code, Dataframe):
        self.code = Power_Code
        self.dataframe = Dataframe
        self.power_volume = set_item(Power_Code, Dataframe)[0]  # 전기용량
        self.real_array = set_item(Power_Code, Dataframe)[1]   # 31 * 24 array임.
        self.pred_array = set_item(Power_Code, Dataframe)[2]   # 31 * 24 array임
        self.usage_masked = usage_masking(self.real_array / self.power_volume) 
        self.error_array = error_cal(self.power_volume, self.real_array, self.pred_array)
        self.error_masked = error_masking(error_cal(self.power_volume, self.real_array, self.pred_array))
        self.real_money = self.real_array * self.usage_masked * self.error_masked
        self.best_money = self.real_array * 4
        self.true_money  = self.best_money - self.real_money
   
def damage_sum(item_list, Dataframe):
    total_real_array = np.zeros((31, 24))
    total_pred_array = np.zeros((31, 24))
    total_power_volumes = 0

    for i in item_list:
        temp = Processing(i, Dataframe)
        total_real_array = total_real_array + temp.real_array
        total_pred_array = total_pred_array + temp.pred_array
        total_power_volumes = total_power_volumes + temp.power_volume
        
    total_usage_masked = usage_masking(total_real_array / total_power_volumes) 
    total_error_array = error_cal(total_power_volumes, total_real_array, total_pred_array)
    total_error_masked = error_masking(error_cal(total_power_volumes, total_real_array, total_pred_array))
    total_real_money = total_real_array * total_usage_masked * total_error_masked
    total_best_money = total_real_array * 4
    total_true_money  = total_best_money - total_real_money
    
    return total_true_money.sum()

def damage_percent(item_list, Dataframe):
    total_real_array = np.zeros((31, 24))
    total_pred_array = np.zeros((31, 24))
    total_power_volumes = 0

    for i in item_list:
        temp = Processing(i, Dataframe)
        total_real_array = total_real_array + temp.real_array
        total_pred_array = total_pred_array + temp.pred_array
        total_power_volumes = total_power_volumes + temp.power_volume
        
    total_usage_masked = usage_masking(total_real_array / total_power_volumes) 
    total_error_array = error_cal(total_power_volumes, total_real_array, total_pred_array)
    total_error_masked = error_masking(error_cal(total_power_volumes, total_real_array, total_pred_array))
    total_real_money = total_real_array * total_usage_masked * total_error_masked
    total_best_money = total_real_array * 4
    total_take_percent = (total_real_money.sum()) / (total_best_money.sum())
    
    return total_take_percent

def volume_checker(item_list, newitem, Dataframe, min_length, maxvolume):
    medium_volume = 0
    small_volume = 0
    newitem_volume = set_item(newitem, Dataframe)[0]
    length_item = len(item_list)
    
    checker_1 = False
    checker_2 = False
    
    for i in item_list:
        power_volume = set_item(i, Dataframe)[0]
        
        if power_volume > 1000:
            medium_volume = medium_volume + power_volume
        else:
            small_volume = small_volume + power_volume
    
    total_volume = medium_volume + small_volume
    
    if length_item < min_length:
        checker_1 = True
    else:
        if medium_volume < small_volume:
            checker_1 = True
        else:
            if newitem_volume <= 1000:
                checker_1 = True
        
    if total_volume < maxvolume:
        checker_2 = True
        
    return checker_1, checker_2

def sampler(total_component, item_list, Dataframe):
      random.shuffle(item_list)
      item = item_list[0]

      temp_tuples = []
      for index, component in enumerate(total_component):
            temp_component = component.copy()
            temp_component.append(item)
            damage_now = damage_sum(component, Dataframe)
            damage_pred = damage_sum(temp_component, Dataframe)
            
            if damage_pred < damage_now:
                  change = damage_now - damage_pred
                  temp_tuple = (index, change)
                  temp_tuples.append(temp_tuple)
            
      if len(temp_tuples) > 0:
            max_value = max(temp_tuples, key=lambda x: x[1])
            max_index = temp_tuples.index(max_value)
      
            if max_value[1] > 0:
                  total_component[max_index].append(item)
                  item_list.remove(item)
                  
def sampler_rate(total_component, item_list, min_length, max_volume, Dataframe):
      random.shuffle(item_list)
      item = item_list[0]
      temp_tuples = []
      for index, component in enumerate(total_component):
            temp_component = component.copy()
            temp_component.append(item)
            damage_now = damage_percent(component, Dataframe)
            damage_pred = damage_percent(temp_component, Dataframe)
            
            check_1, check_2 = volume_checker(temp_component, item, Dataframe, min_length, max_volume)
            
            if (check_1 == True) & (check_2 == True):   
                  change = damage_now - damage_pred
                  temp_tuple = (index, change)
                  temp_tuples.append(temp_tuple)
            
      min_value = min(temp_tuples, key=lambda x: x[1])
      min_index = temp_tuples.index(min_value)
      total_component[min_index].append(item)
      item_list.remove(item)