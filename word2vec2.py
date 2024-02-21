import numpy as np
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/nlpl_82", filename="model.bin"), binary=True, unicode_errors="ignore")

#ベクトル演算予測
def cos(model):
    result = model.most_similar(positive=['madrid', 'france'], negative=['spain'])

    # 類似度の高い順にソート
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    # 上位5つを表示
    for i, (word, similarity) in enumerate(sorted_result[:5], 1):
        print(f"{i}. {word}: {similarity:.4f}")
    print("\n")

cos(model)


# 埋め込み層の取得
embedding_layer = model

'''
#単語がvocabularyに存在するか確認
try:
    vector = model['france']
    # ここでベクトルを使用する処理を実行
except KeyError:
    print("Word not in vocabulary")
'''

# 単語の取得
with open('data/country2.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

# 複数の単語の埋め込みベクトルを取得
embeddings = [embedding_layer[word] for word in words]

# numPyの配列に変換
embeddings = np.array(embeddings)

word_size = int(len(words)/2)

cos_sim_matrix = np.zeros((word_size,word_size))

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


for i in range(word_size):
    for j in range(word_size):
        v1 = embeddings[2*i] - embeddings[2*i+1]
        v2 = embeddings[2*j] - embeddings[2*j+1]

        cos_sim_matrix[i,j] = format(cos_sim(v1,v2),'.2f')

print(cos_sim_matrix)

matrix1 = []
for i in range(0, word_size - 1):
    for j in range(i+1, word_size):
        matrix1.append(cos_sim_matrix[i,j])

matrix1 = np.array(matrix1)

ave = np.mean(matrix1)
var = np.var(matrix1)

ave = round(ave, 3)
var = round(var, 3)

print(f"{ave=}")
print(f"{var=}")

#t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib

# t-SNEで2次元に変換
tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_embeddings = tsne_model.fit_transform(embeddings)

'''
# 2次元に変換されたベクトルの表示
for word, vector in zip(words, tsne_embeddings):
    print(f'{word}: {vector}')
'''

# プロット
plt.figure(figsize=(8, 8))
for word, vector in zip(words, tsne_embeddings):
    plt.scatter(vector[0], vector[1])
    plt.annotate(word, (vector[0], vector[1]), alpha=0.7)

# x軸に線を引く
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
# y軸に線を引く
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)

plt.title('t-SNE Visualization of Word Embeddings')
plt.show()
