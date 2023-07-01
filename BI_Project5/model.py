import random
from util import Processing, sampler, sampler_rate

class Kcluster:
    def __init__(self, power_list, Dataframe):
        self.total_power_list = power_list
        self.dataframe = Dataframe

    def start_set(self, numk, method = 'high'):
        K_Number = numk
        total_component = []
        for num in range(K_Number):
            temp_list = []
            total_component.append(temp_list)

        if method == 'high':
            damage_dict = dict()
            power_list = self.total_power_list
            for i in power_list:
                temp = Processing(i, self.dataframe)
                sum_damage = temp.true_money.sum()
                damage_dict[i] = sum_damage

            del_list = []
            sorted_dict = sorted(damage_dict.items(), key=lambda x: x[1], reverse=True)
            for order, values in enumerate(sorted_dict[:K_Number]):
                total_component[order].append(values[0])
                del_list.append(values[0])
            for delete in del_list:
                power_list.remove(delete)

            return total_component, power_list

        if method == 'random':
            damage_dict = dict()
            item_list = self.total_power_list
            random.shuffle(item_list)
            
            for i in item_list[:K_Number]:
                temp = Processing(i, self.dataframe)
                sum_damage = temp.true_money.sum()
                damage_dict[i] = sum_damage

            del_list =[]
            for order, values in enumerate(damage_dict):
                print(values)
                total_component[order].append(values)
                del_list.append(values)
            for delete in del_list:
                print(delete)
                item_list.remove(delete)
            return total_component, item_list
        
def Let_model(power_list, model, k, first_search, min_length, max_volume, Dataframe):
    Power_list = power_list
    model = model
    
    total_component, item_list = model.start_set(k, 'high')
    print(f'Model Start. k = {k}, Strict Search Range = {first_search}, Min Component Size = {min_length}, Max Volume = {max_volume}')
    
    for i in range(first_search):
        sampler(total_component, item_list, Dataframe)
        
    while len(item_list) != 0:
        sampler_rate(total_component, item_list, min_length, max_volume, Dataframe)
        
    return total_component