from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib

# モデルのロード
model_path = '/Users/nakanishitakumi/Downloads/latest-ja-word2vec-gensim-model/word2vec.gensim.model'
model = Word2Vec.load(model_path)

# 埋め込み層の取得
embedding_layer = model.wv

# 単語の取得
with open('data/data1.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

# 複数の単語の埋め込みベクトルを取得
embedding_vectors = [embedding_layer[word] for word in words]

# NumPyの配列に変換
embedding_vectors = np.array(embedding_vectors)

word_size = int(len(words)/2)

cos_sim_matrix = np.zeros((word_size,word_size))

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

for i in range(word_size):
    for j in range(word_size):
        v1 = embedding_vectors[2*i] - embedding_vectors[2*i+1]
        v2 = embedding_vectors[2*j] - embedding_vectors[2*j+1]

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


'''
# t-SNEで2次元に変換
tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_embeddings = tsne_model.fit_transform(embedding_vectors)


# 2次元に変換されたベクトルの表示
for word, vector in zip(words, tsne_embeddings):
    print(f'{word}: {vector}')

    
num_words = len(words)

for i in range(0, num_words-1, 2):
    point1 = tsne_embeddings[i]
    point2 = tsne_embeddings[i + 1]

    a = (point1[1] - point2[1])/(point1[0] - point2[0])
    print(f'{words[i]}と{words[i+1]}の傾き: {a}')

# プロット
plt.figure(figsize=(8, 8))
for word, vector in zip(words, tsne_embeddings):
    plt.scatter(vector[0], vector[1])
    plt.annotate(word, (vector[0], vector[1]), alpha=0.7)

plt.title('t-SNE Visualization of Word Embeddings')
plt.show()
'''