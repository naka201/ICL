import openai
import pandas as pd

# OpenAI APIキーを設定
api_key = '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'
openai.api_key = api_key

# 単語の取得
with open('data/pred2.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

# 各単語の埋め込みをOpenAI APIから取得
embeddings = []
for word in words:
    result = openai.Embedding.create(model="text-embedding-3-large", input=word)
    embedding = result.data[0].embedding
    embeddings.append(embedding)

# データフレームの作成
df = pd.DataFrame(embeddings, columns=[f"dimension_{i}" for i in range(len(embeddings[0]))])
df['Word'] = words

# 結果を表示
print(df)

# CSVファイルにデータフレームを保存
df.to_csv('dataset/dataset_variouspred.csv', index=False)
