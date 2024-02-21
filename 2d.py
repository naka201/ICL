import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.optim.lr_scheduler import LambdaLR
#from sklearn.linear_model import LinearRegression

# データの読み込み
data1 = pd.read_csv('dataframe_openai.csv')
data2 = pd.read_csv('dataframe_pred.csv')

data3 = pd.read_csv('dataframe_learn1.csv')
data4 = pd.read_csv('dataframe_learn2.csv')

data5 = pd.read_csv('learn_dim_2.csv')
data6 = pd.read_csv('pred_dim_2.csv')

# 単語と対応する埋め込みベクトルを取得
word_embeddings = data5.set_index('Word').to_dict(orient='index')
pred_embeddings = data6.set_index('Word').to_dict(orient='index')

def torch_fix_seed(seed=42):
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()

class LinearRegression(nn.Module):
    def __init__(self):
        #super()を用いて継承元のコンストラクタを呼び出す
        super().__init__()

        self.theta = nn.Parameter(torch.rand(size=(3,)))

    """
    順伝播関数
    Moduleクラスのインスタンスは__call__が定義されており、呼び出すとforwardメソッドが呼び出され、順伝播が行われる。
    x: 入力データ
    """
    def forward(self, ca1, co1, co2):
        # 回転行列の定義
        rotation1 = torch.tensor([[torch.cos(self.theta[0]), -torch.sin(self.theta[0])],
                                  [torch.sin(self.theta[0]), torch.cos(self.theta[0])]])
        
        rotation2 = torch.tensor([[torch.cos(self.theta[1]), -torch.sin(self.theta[1])],
                                  [torch.sin(self.theta[1]), torch.cos(self.theta[1])]])
        
        rotation3 = torch.tensor([[torch.cos(self.theta[2]), -torch.sin(self.theta[2])],
                                  [torch.sin(self.theta[2]), torch.cos(self.theta[2])]])

        # 行列の掛け算を行う
        result_vector = torch.matmul(rotation1, ca1)
        result_vector = result_vector - torch.matmul(rotation2, co1)
        result_vector = result_vector + torch.matmul(rotation3, co2)
        return result_vector

import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

# 学習中に損失を記録するためのリストを用意
losses = []

def learn():
    # 学習の試行回数
    num_trials = 8000
    batch_size = 20

    # 線形回帰モデルを作成
    model = LinearRegression()
    #criterion = nn.MSELoss()
    criterion = nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    #alpha = 0.0001 #L2正則化のパラメータ
    alpha = 0.001 #損失関数がコサイン類似度のとき用のL2正則化のパラメータ

    # 学習率スケジューラを定義
    step_size = 20
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch))

    for epoch in range(num_trials):
        batch_co1 = []
        batch_ca1 = []
        batch_co2 = []
        batch_ca2 = []

        for _ in range(batch_size):
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
            batch_co1.append(list(word_embeddings[co1].values()))
            batch_ca1.append(list(word_embeddings[ca1].values()))
            batch_co2.append(list(word_embeddings[co2].values()))
            batch_ca2.append(list(word_embeddings[ca2].values()))

        ret = model(torch.tensor(batch_ca1), torch.tensor(batch_co1), torch.tensor(batch_co2))
        #loss = criterion(ret, torch.tensor(batch_ca2))
        loss = (-1) * criterion(ret, torch.tensor(batch_ca2)).mean()

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % step_size == 0:
            # 学習率スケジューラを更新
            scheduler.step()

        if epoch % 1000 == 0:
            # 重みと切片を出力
            weights = torch.nn.utils.parameters_to_vector(model.parameters())
            print(f"エポック {epoch}: 重み: {weights}")

    # 損失の変化をプロット
    plot_loss(losses)

    return model


#予測
def pred(model):
    for i in range(10):
        # 1つ目のセットのindexを選択
        index1 = np.random.choice(np.arange(0, 5, 2))
        index2 = index1 + 1

        # 選ばれた単語の取得
        co1 = list(pred_embeddings.keys())[index1]
        ca1 = list(pred_embeddings.keys())[index2]

        # 2つ目のセットのindexを選択
        available_indices = list(set(np.arange(0, 5, 2)) - {index1, index2})
        index3 = np.random.choice(available_indices)
        #index4 = index3 + 1
        index4 = np.random.choice(np.arange(0, 5, 1))

        # 選ばれた単語の取得
        co2 = list(pred_embeddings.keys())[index3]
        ca2 = list(pred_embeddings.keys())[index4]
    
        
        # 選ばれた単語に対応する埋め込みベクトルの取得
        country1 = torch.tensor(list(pred_embeddings[co1].values()))
        capital1 = torch.tensor(list(pred_embeddings[ca1].values()))
        country2 = torch.tensor(list(pred_embeddings[co2].values()))
        capital2 = torch.tensor(list(pred_embeddings[ca2].values()))
        
        # 平均重みを使用して予測
        predicted_capital2 = model(capital1, country1, country2).detach().numpy()

        # 予測結果を表示
        print(f"{ca1} - {co1} + {co2} = {ca2}")
        print("予測された首都ベクトル:", predicted_capital2)
        print("想定解のベクトル : ", capital2)

        from sklearn.metrics.pairwise import cosine_similarity

        # コサイン類似度を計算
        cosine_sim = cosine_similarity(predicted_capital2.reshape(1, -1), capital2.reshape(1, -1))

        # 結果を表示
        print("コサイン類似度:", cosine_sim[0, 0])
        print("------------------------------------")

from sklearn.metrics.pairwise import cosine_similarity

def pred2(model):
    for i in range(40, 50, 2):
        co1 = list(pred_embeddings.keys())[i]
        ca1 = list(pred_embeddings.keys())[i+1]
        country1 = torch.tensor(list(pred_embeddings[co1].values()))
        capital1 = torch.tensor(list(pred_embeddings[ca1].values()))
        
        if i != 40:
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
        if i != 48:
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
