import openai
import numpy as np


# APIキーを設定
api_key = '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'
openai.api_key = api_key


with open('data/country.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]


embeddings2 = []
for word in words:
    embeddings2.append(openai.Embedding.create(model="text-embedding-3-large", input=word).data[0].embedding)


for i in range(len(embeddings2)):
    size = np.array(embeddings2[i])
    print(np.linalg.norm(size))





import matplotlib.pyplot as plt

# embeddings2 リストから x 座標と y 座標を取得
x_coords = [embedding[0] for embedding in embeddings2]
y_coords = [embedding[1] for embedding in embeddings2]

# 散布図をプロット
plt.scatter(x_coords, y_coords)

# 各点に対応する単語をテキストで表示
for i, word in enumerate(words):
    plt.annotate(word, (x_coords[i], y_coords[i]))

# グラフを表示
#plt.show()