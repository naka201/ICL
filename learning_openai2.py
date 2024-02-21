import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# データセットの読み込み（例）
data = pd.read_csv('dataframe_openai.csv')

# 単語と対応する埋め込みベクトルを取得
word_embeddings = data.set_index('Word').iloc[:, :-1].to_dict(orient='index')

df = pd.DataFrame(data)

def emb_get(word_list, df):
    vec_list = []
    # 特定の単語に対応する埋め込みベクトルの取得
    for word in word_list:
        embedding_vector = df[df['Word'] == word].iloc[:, :-1].values.flatten()
        vec_list.append(embedding_vector)
    return np.vstack(vec_list)

capital1 = []
country1 = []
country2 = []
capital2 = []
for i in range(len(data.Word)):
    if i % 2 == 0:
        country1.append(data.Word[i])
        country2.append(data.Word[i])
    else:
        capital1.append(data.Word[i])
        capital2.append(data.Word[i])

# 全組み合わせの特徴量行列 X, 目的変数行列 Y の作成
X_all = np.hstack([emb_get(capital1, df), emb_get(country1, df), emb_get(country2, df)])
Y_all = emb_get(capital2, df)

# 同じインデックスの組み合わせだけを抽出
index_mask1 = np.arange(len(country1)) == np.arange(len(capital1))
X_selected = X_all[index_mask1]
Y_selected = Y_all[index_mask1]

# 同じインデックスの組み合わせだけを抽出
index_mask2 = np.arange(len(country2)) == np.arange(len(capital2))
X_selected = X_selected[index_mask2]
Y_selected = Y_selected[index_mask2]

print(X_selected)

# モデルの作成と学習
model = LinearRegression()
model.fit(X_selected, Y_selected)

# 学習結果の表示
print("学習結果:")
print("重み行列:")
print(model.coef_)
print("切片:")
print(model.intercept_)


#予測
def pred(learn_from_weigths):
    for i in range(5):
        # 1つ目のセットのindexを選択
        index1 = np.random.choice(np.arange(0, 9, 2))
        index2 = index1 + 1

        # 選ばれた単語の取得
        country1 = list(word_embeddings.keys())[index1]
        capital1 = list(word_embeddings.keys())[index2]

        # 2つ目のセットのindexを選択
        available_indices = list(set(np.arange(0, 9, 2)) - {index1, index2})
        index3 = np.random.choice(available_indices)
        index4 = index3 + 1

        # 選ばれた単語の取得
        country2 = list(word_embeddings.keys())[index3]
        capital2 = list(word_embeddings.keys())[index4]

        # 選ばれた単語に対応する埋め込みベクトルの取得
        country1_vector = np.array(list(word_embeddings[country1].values()))
        capital1_vector = np.array(list(word_embeddings[capital1].values()))
        country2_vector = np.array(list(word_embeddings[country2].values()))
        capital2_vector = np.array(list(word_embeddings[capital2].values()))

        # 新しいデータの特徴量行列を作成
        new_X = np.column_stack((capital1_vector, country1_vector, country2_vector))

        # 平均重みを使用して予測
        predicted_capital2_vector = np.dot(new_X, learn_from_weigths)

        # 予測結果を表示
        print(f"{capital1} - {country1} + {country2} = {capital2}")
        print("予測された首都ベクトル:", predicted_capital2_vector)

        from sklearn.metrics.pairwise import cosine_similarity

        # コサイン類似度を計算
        cosine_sim = cosine_similarity(predicted_capital2_vector.reshape(1, -1), capital2_vector.reshape(1, -1))

        # 結果を表示
        print("コサイン類似度:", cosine_sim[0, 0])
        print("------------------------------------")

def main():
    pred(model.coef_)

if __name__ == "__main__":
    main()