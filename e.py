import pandas as pd
import numpy as np
from torch import nn
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression

data5 = pd.read_csv('learn_dim_2.csv')
data6 = pd.read_csv('pred_dim_2.csv')

# 単語と対応する埋め込みベクトルを取得
word_embeddings = data5.set_index('Word').to_dict(orient='index')
pred_embeddings = data6.set_index('Word').to_dict(orient='index')


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(size=(4,)))
        self.theta = nn.Parameter(torch.rand(size=(3,), requires_grad=True))
        
    def forward(self, ca1, co1, co2):
        # 回転行列の定義
        rotation1 = torch.tensor([[torch.cos(self.theta[0]), -torch.sin(self.theta[0])],
                                  [torch.sin(self.theta[0]), torch.cos(self.theta[0])]], dtype=torch.float32)
        
        rotation2 = torch.tensor([[torch.cos(self.theta[1]), -torch.sin(self.theta[1])],
                                  [torch.sin(self.theta[1]), torch.cos(self.theta[1])]],dtype=torch.float32)
        
        rotation3 = torch.tensor([[torch.cos(self.theta[2]), -torch.sin(self.theta[2])],
                                  [torch.sin(self.theta[2]), torch.cos(self.theta[2])]],dtype=torch.float32)
        
    
        # 行列の掛け算を行う
        return self.w[0] * torch.matmul(rotation1, ca1) - self.w[1] * torch.matmul(rotation2, co1) + self.w[2] * torch.matmul(rotation3, co2) + self.w[3]
    

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

def learn():
    # 学習の試行回数
    num_trials = 500

    # 線形回帰モデルを作成
    model = LinearRegression()
    criterion = nn.MSELoss()
    #criterion = torch.nn.CosineSimilarity(dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    alpha = 0.001

    # スケジューラの設定
    step_size = 20
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch))

    losses = []  # 各エポックでの損失を保存するリスト

    for epoch in range(num_trials):
        # 1つ目のセットのindexを選択
        index1 = np.random.choice(np.arange(0, 50, 2))
        index2 = index1 + 1
        # 2つ目のセットのindexを選択
        available_indices = list(set(np.arange(0, 50, 2)) - {index1, index2})
        index3 = np.random.choice(available_indices)
        index4 = index3 + 1

        # 選ばれた単語の取得
        co1 = list(word_embeddings.keys())[index1]
        ca1 = list(word_embeddings.keys())[index2]
        # 選ばれた単語の取得
        co2 = list(word_embeddings.keys())[index3]
        ca2 = list(word_embeddings.keys())[index4]

        # 選ばれた単語に対応する埋め込みベクトルの取得
        country1 = torch.tensor(list(word_embeddings[co1].values()), requires_grad=True)
        capital1 = torch.tensor(list(word_embeddings[ca1].values()), requires_grad=True)
        country2 = torch.tensor(list(word_embeddings[co2].values()), requires_grad=True)
        capital2 = torch.tensor(list(word_embeddings[ca2].values()), requires_grad=True)

        ret = model(capital1, country1, country2)
        loss = criterion(ret, capital2)
        #loss = (-1) * criterion(ret, capital2)

        # パラメータのL2ノルムの二乗を損失関数に足す
        l2 = torch.tensor(0., requires_grad=True)
        for w in model.parameters():
            l2 = l2 + torch.norm(w)**2
        loss = loss + alpha*l2
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % step_size == 0:
            # 学習率スケジューラを更新
            scheduler.step()

        if epoch % 10 == 0:
            # 重みと切片を出力
            weights = torch.nn.utils.parameters_to_vector(model.parameters())
            print(f"エポック {epoch}: 重み: {weights}")

        losses.append(loss.item())  # 損失を保存
    
    plot_loss(losses)

    return model


from sklearn.metrics.pairwise import cosine_similarity

#予測
def pred2(model):
    for i in range(0, 10, 2):
        co1 = list(pred_embeddings.keys())[i]
        ca1 = list(pred_embeddings.keys())[i+1]
        country1 = torch.tensor(list(pred_embeddings[co1].values()))
        capital1 = torch.tensor(list(pred_embeddings[ca1].values()))
        
        if i != 0:
            for j in range(0, i, 2):
                co2 = list(pred_embeddings.keys())[j]
                country2 = torch.tensor(list(pred_embeddings[co2].values()))
                predicted_capital2 = model(capital1, country1, country2).detach().numpy()

                for k in range(1, 50, 2):
                    ca2 = list(pred_embeddings.keys())[k]
                    capital2 = torch.tensor(list(pred_embeddings[ca2].values()))

                    cosine_sim = cosine_similarity(predicted_capital2.reshape(1, -1), capital2.reshape(1, -1))
                    print(f"{ca1} - {co1} + {co2} = {ca2}? : {cosine_sim}")
                print("------------------------------------")
        if i != 8:
            for j in range(i, 10, 2):
                co2 = list(pred_embeddings.keys())[j]
                country2 = torch.tensor(list(pred_embeddings[co2].values()))
                predicted_capital2 = model(capital1, country1, country2).detach().numpy()

                for k in range(1, 50, 2):
                    ca2 = list(pred_embeddings.keys())[k]
                    capital2 = torch.tensor(list(pred_embeddings[ca2].values()))

                    cosine_sim = cosine_similarity(predicted_capital2.reshape(1, -1), capital2.reshape(1, -1))
                    print(f"{ca1} - {co1} + {co2} = {ca2}? : {cosine_sim}")

                print("------------------------------------")

        

def main():
    result_from_learn = learn()
    pred2(result_from_learn)

if __name__ == "__main__":
    main()
