import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# データの読み込み
data = pd.read_csv('dataframe_openai.csv')
data2 = pd.read_csv('dataframe_openai4.csv')

# 単語と対応する埋め込みベクトルを取得
word_embeddings = data.set_index('Word').iloc[:, :-1].to_dict(orient='index')
pred_embeddings = data2.set_index('Word').iloc[:, :-1].to_dict(orient='index')

print(data2)

def learn():
    # 学習の試行回数
    num_trials = 100  # 例として5回試行

    weights_list = []
    X = []
    target_vector = []

    for _ in range(num_trials):
        # 1つ目のセットのindexを選択
        index1 = np.random.choice(np.arange(0, 15, 2))
        index2 = index1 + 1
        # 2つ目のセットのindexを選択
        available_indices = list(set(np.arange(0, 15, 2)) - {index1, index2})
        index3 = np.random.choice(available_indices)
        index4 = index3 + 1

        # 選ばれた単語の取得
        country1 = list(word_embeddings.keys())[index1]
        capital1 = list(word_embeddings.keys())[index2]
        # 選ばれた単語の取得
        country2 = list(word_embeddings.keys())[index3]
        capital2 = list(word_embeddings.keys())[index4]

        # 選ばれた単語に対応する埋め込みベクトルの取得
        country1_vector = np.array(list(word_embeddings[country1].values()))
        capital1_vector = np.array(list(word_embeddings[capital1].values()))
        country2_vector = np.array(list(word_embeddings[country2].values()))
        capital2_vector = np.array(list(word_embeddings[capital2].values()))

        # 特徴量行列を作成
        X.append(np.column_stack((capital1_vector, -country1_vector, country2_vector)))

        # 目標ベクトルの作成
        target_vector.append(capital2_vector)

    # 線形回帰モデルを作成
    model = LinearRegression()

    # モデルにデータを適合させる
    model.fit(X, target_vector)

    # 重みと切片を出力
    weights = model.coef_
    intercept = model.intercept_

    print(f"選ばれた単語セット: {capital1} - {country1} + {country2} = {capital2}")
    print("重み:", weights)
    print("切片:", intercept)
    print("-----")

    print(f"model's score : {model.score(X, target_vector)}")
        
    # 重みを保存
    weights_list.append(model.coef_)

    # 重みの平均を計算
    average_weights = np.mean(weights_list, axis=0)

    print("平均重み:", average_weights)

    return average_weights

#予測
def pred(learn_from_weigths):
    for i in range(5):
        # 1つ目のセットのindexを選択
        index1 = np.random.choice(np.arange(0, 5, 2))
        index2 = index1 + 1

        # 選ばれた単語の取得
        country1 = list(pred_embeddings.keys())[index1]
        capital1 = list(pred_embeddings.keys())[index2]

        # 2つ目のセットのindexを選択
        available_indices = list(set(np.arange(0, 5, 2)) - {index1, index2})
        index3 = np.random.choice(available_indices)
        index4 = index3 + 1

        # 選ばれた単語の取得
        country2 = list(pred_embeddings.keys())[index3]
        capital2 = list(pred_embeddings.keys())[index4]

        # 選ばれた単語に対応する埋め込みベクトルの取得
        country1_vector = np.array(list(pred_embeddings[country1].values()))
        capital1_vector = np.array(list(pred_embeddings[capital1].values()))
        country2_vector = np.array(list(pred_embeddings[country2].values()))
        capital2_vector = np.array(list(pred_embeddings[capital2].values()))

        # 新しいデータの特徴量行列を作成
        new_X = np.column_stack((capital1_vector, -country1_vector, country2_vector))

        # 平均重みを使用して予測
        predicted_capital2_vector = np.dot(new_X, learn_from_weigths)

        # 予測結果を表示
        print(f"{capital1} - {country1} + {country2} = {capital2}")
        print("予測された首都ベクトル:", predicted_capital2_vector)
        print("想定解のベクトル : ", capital2_vector)

        from sklearn.metrics.pairwise import cosine_similarity

        # コサイン類似度を計算
        cosine_sim = cosine_similarity(predicted_capital2_vector.reshape(1, -1), capital2_vector.reshape(1, -1))

        # 結果を表示
        print("コサイン類似度:", cosine_sim[0, 0])
        print("------------------------------------")

        

def main():
    result_from_learn = learn()
    pred(result_from_learn)

if __name__ == "__main__":
    main()
