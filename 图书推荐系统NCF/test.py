import numpy as np
import pandas as pd
import random
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 全局参数，随机种子，图像尺寸
seed = 114514
np.random.seed(seed)
random.seed(seed)
BATCH_SIZE = 512

hidden_dim = 16
epochs = 1
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

df = pd.read_csv('train_dataset.csv')
print('共{}个用户，{}本图书，{}条记录'.format(max(df['user_id']) + 1, max(df['item_id']) + 1, len(df)))


class Goodbooks(Dataset):
    def __init__(self, df, mode='training', negs=99):
        super().__init__()

        self.df = df
        self.mode = mode

        self.book_nums = max(df['item_id']) + 1
        self.user_nums = max(df['user_id']) + 1

        self._init_dataset()

    def _init_dataset(self):
        self.Xs = []

        self.user_book_map = {}
        for i in range(self.user_nums):
            self.user_book_map[i] = []

        for index, row in self.df.iterrows():
            user_id, book_id = row
            self.user_book_map[user_id].append(book_id)

        if self.mode == 'training':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                for item in items[:-1]:
                    self.Xs.append((user, item, 1))
                    for _ in range(3):
                        while True:
                            neg_sample = random.randint(0, self.book_nums - 1)
                            if neg_sample not in self.user_book_map[user]:
                                self.Xs.append((user, neg_sample, 0))
                                break

        elif self.mode == 'validation':
            for user, items in tqdm.tqdm(self.user_book_map.items()):
                if len(items) == 0:
                    continue
                self.Xs.append((user, items[-1]))

    def __getitem__(self, index):

        if self.mode == 'training':
            user_id, book_id, label = self.Xs[index]
            return user_id, book_id, label
        elif self.mode == 'validation':
            user_id, book_id = self.Xs[index]

            negs = list(random.sample(
                list(set(range(self.book_nums)) - set(self.user_book_map[user_id])),
                k=99
            ))
            return user_id, book_id, torch.LongTensor(negs)

    def __len__(self):
        return len(self.Xs)


# 建立训练和验证dataloader
traindataset = Goodbooks(df, 'training')
validdataset = Goodbooks(df, 'validation')

trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
validloader = DataLoader(validdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)


# 构建模型
class NCFModel(torch.nn.Module):
    def __init__(self, hidden_dim, user_num, item_num, mlp_layer_num=4, weight_decay=1e-5, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_layer_num = mlp_layer_num
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.mlp_user_embedding = torch.nn.Embedding(user_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))
        self.mlp_item_embedding = torch.nn.Embedding(item_num, hidden_dim * (2 ** (self.mlp_layer_num - 1)))

        self.gmf_user_embedding = torch.nn.Embedding(user_num, hidden_dim)
        self.gmf_item_embedding = torch.nn.Embedding(item_num, hidden_dim)

        mlp_Layers = []
        input_size = int(hidden_dim * (2 ** (self.mlp_layer_num)))
        for i in range(self.mlp_layer_num):
            mlp_Layers.append(torch.nn.Linear(int(input_size), int(input_size / 2)))
            mlp_Layers.append(torch.nn.Dropout(self.dropout))
            mlp_Layers.append(torch.nn.ReLU())
            input_size /= 2
        self.mlp_layers = torch.nn.Sequential(*mlp_Layers)

        self.output_layer = torch.nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, user, item):
        user_gmf_embedding = self.gmf_user_embedding(user)
        item_gmf_embedding = self.gmf_item_embedding(item)

        user_mlp_embedding = self.mlp_user_embedding(user)
        item_mlp_embedding = self.mlp_item_embedding(item)

        gmf_output = user_gmf_embedding * item_gmf_embedding

        mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))).squeeze(-1)

        # return -r_pos_neg + reg
        return output

    def predict(self, user, item):
        self.eval()
        with torch.no_grad():
            user_gmf_embedding = self.gmf_user_embedding(user)
            item_gmf_embedding = self.gmf_item_embedding(item)

            user_mlp_embedding = self.mlp_user_embedding(user)
            item_mlp_embedding = self.mlp_item_embedding(item)

            gmf_output = user_gmf_embedding.unsqueeze(1) * item_gmf_embedding

            user_mlp_embedding = user_mlp_embedding.unsqueeze(1).expand(-1, item_mlp_embedding.shape[1], -1)
            mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
            mlp_output = self.mlp_layers(mlp_input)

        output = torch.sigmoid(self.output_layer(torch.cat([gmf_output, mlp_output], dim=-1))).squeeze(-1)
        return output


model = NCFModel(hidden_dim, traindataset.user_nums, traindataset.book_nums).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.BCELoss()

loss_for_plot = []
hits_for_plot = []

for epoch in range(epochs):

    losses = []
    for index, data in enumerate(trainloader):
        user, item, label = data
        user, item, label = user.to(device), item.to(device), label.to(device).float()
        y_ = model(user, item).squeeze()

        loss = crit(y_, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    hits = []
    for index, data in enumerate(validloader):
        user, pos, neg = data
        pos = pos.unsqueeze(1)
        all_data = torch.cat([pos, neg], dim=-1)
        output = model.predict(user.to(device), all_data.to(device)).detach().cpu()

        for batch in output:
            if 0 not in (-batch).argsort()[:10]:
                hits.append(0)
            else:
                hits.append(1)
    print('Epoch {} finished, average loss {}, hits@20 {}'.format(epoch, sum(losses) / len(losses),
                                                                  sum(hits) / len(hits)))
    loss_for_plot.append(sum(losses) / len(losses))
    hits_for_plot.append(sum(hits) / len(hits))

# 模型保存
torch.save(model.state_dict(), 'model.h5')

import matplotlib.pyplot as plt

x = list(range(1, len(hits_for_plot) + 1))
plt.subplot(1, 2, 1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x, loss_for_plot, 'r')

plt.subplot(1, 2, 2)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.plot(x, hits_for_plot, 'r')

plt.show()

df = pd.read_csv('test_dataset.csv')
user_for_test = df['user_id'].tolist()

predict_item_id = []


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i, i + n]


f = open('submission.csv', 'w', encoding='utf-8')

for user in user_for_test:
    # 将用户已经交互过的物品排除
    user_visited_items = traindataset.user_book_map[user]
    items_for_predict = list(set(range(traindataset.book_nums)) - set(user_visited_items))

    results = []
    user = torch.Tensor([user]).to(device)

    for batch in chunks(items_for_predict, 64):
        batch = torch.Tensor(batch).unsqueeze(0).to(device)

        result = model(user, batch).view(-1).detach().cpu()
        results.append(result)

    results = torch.cat(results, dim=-1)
    predict_item_id = (-results).argsort()[:10]
    list(map(lambda x: f.write('{},{}\n'.format(user.cpu().item(), x)), predict_item_id))

f.flush()
f.close()
