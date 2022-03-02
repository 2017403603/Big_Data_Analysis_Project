import pandas as pd
import os


data = pd.read_csv('./data/train_dataset.csv', encoding='utf-8')
# data['user_id']=data['user_id']+1
# data['item_id']=data['item_id']+1
print(data)
data.to_csv('./data/ml-1m.txt', index=0, header=0,encoding='utf-8')