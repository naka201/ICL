import pandas as pd
import numpy as np


import openai
import pandas as pd

# OpenAI APIキーを設定
api_key = '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'
openai.api_key = api_key

# 単語の取得
with open('data/country3.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

embeddings = []
for word in words:
    embeddings.append(openai.Embedding.create(model="text-embedding-ada-002", input=word).data[0].embedding)

# データセットの作成
for i in range(0, 10, 2):
    for j in range(0, 10, 2):
        dataset_i = (np.stack([words[i+1], words[i], words[j]]), words[j+1])

# 特徴量行列と目的変数の取得
X_1, y_1 = dataset_1
X_2, y_2 = dataset_2

# データフレームに変換
data = pd.DataFrame({
    'Feature_Matrix': [X_1, X_2],
    'Target': [y_1, y_2]
})

# データフレームの表示
print(data)
# 結果を表示
print(df)

# CSVファイルにデータフレームを保存
df.to_csv('dataframe_openai4.csv', index=False)



