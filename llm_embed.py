import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.optim.lr_scheduler import LambdaLR
import argparse


# データの読み込み
data1 = pd.read_csv('dataset/dataset_variousword.csv')
data2 = pd.read_csv('dataset/dataset_variouspred.csv')

# 単語と対応する埋め込みベクトルを取得
word_embeddings = data1.set_index('Word').iloc[:, :-1].to_dict(orient='index')
pred_embeddings = data2.set_index('Word').iloc[:, :-1].to_dict(orient='index')

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

        #線形結合を行う(wx + bとしてw,bがパラメータとして保持される)
        self.w = nn.Parameter(torch.randn(size=(4,)))
        #self.w = nn.Parameter(torch.ones(size=(4,)))


    """
    順伝播関数
    Moduleクラスのインスタンスは__call__が定義されており、呼び出すとforwardメソッドが呼び出され、順伝播が行われる。
    x: 入力データ
    """
    def forward(self, ca1, co1, co2):
        #return self.w[0]*ca1 - self.w[1]*co1 + self.w[2]*co2 + self.w[3]
        return self.w[0]*ca1 - self.w[1]*co1 + self.w[2]*co2 + self.w[3]

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
    batch_size = 100

    # 線形回帰モデルを作成
    model = LinearRegression()
    #criterion = nn.MSELoss()
    criterion = nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    alpha = 0.001 #L2正則化のパラメータ

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

        if epoch % 500 == 0:
            # 重みと切片を出力
            weights = torch.nn.utils.parameters_to_vector(model.parameters())
            print(f"エポック {epoch}: 重み: {weights}")

    # 損失の変化をプロット
    plot_loss(losses)

    torch.save(model.state_dict(), 'saved_model.pth')

from sklearn.metrics.pairwise import cosine_similarity

pred_size = len(pred_embeddings)

def pred():
    similarity_results = []

    model = LinearRegression()
    model.load_state_dict(torch.load('saved_model.pth'))

    for i in range(2, 4, 2):
        co1 = list(pred_embeddings.keys())[i]
        ca1 = list(pred_embeddings.keys())[i+1]
        country1 = torch.tensor(list(pred_embeddings[co1].values()))
        capital1 = torch.tensor(list(pred_embeddings[ca1].values()))
        
        num = 0
        co2 = list(pred_embeddings.keys())[num]
        correct = list(pred_embeddings.keys())[num + 1]

        country2 = torch.tensor(list(pred_embeddings[co2].values()))
        predicted_capital2 = model(capital1, country1, country2).detach().numpy()

        for k in range(0, pred_size):
            ca2 = list(pred_embeddings.keys())[k]
            capital2 = torch.tensor(list(pred_embeddings[ca2].values()))

            cosine_sim = cosine_similarity(predicted_capital2.reshape(1, -1), capital2.reshape(1, -1))
            
            # 結果をリストに追加
            similarity_results.append((ca2, round(cosine_sim[0, 0], 4)))

    # ソート
    sorted_results = sorted(similarity_results, key=lambda x: x[1], reverse=True)

    # ソートした結果を順位とともに出力
    print(f"{ca1} - {co1} + {co2} = {correct}?")
    for rank, (ca2, similarity) in enumerate(sorted_results[:10], start=1):
        print(f"{rank}: {ca2} : {similarity:.4f}")

    # 下位5つを表示
    print(f"\n{ca1} - {co1} + {co2} = {correct}? (Bottom 5)")
    for rank, (ca2, similarity) in enumerate(sorted_results[-5:], start=len(sorted_results)-4):
        print(f"{rank}: {ca2} : {similarity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec", type=str, default="all")

    args = parser.parse_args()

    if args.exec in ["all", "learn_only"]:
        learn()
    if args.exec in ["all", "pred_only"]:
        pred()
