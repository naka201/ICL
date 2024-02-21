from transformers import BertModel, BertTokenizer
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 事前訓練済みのBERTモデルとトークナイザを取得
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# テキストをトークナイズ
text = "Tokyo Japan Madrid Spain Ottawa Canada"
tokens = tokenizer(text, return_tensors="pt")

# モデルにトークンを入力して出力を取得
outputs = model(**tokens)

# 最終層の隠れ層の出力を取得
last_hidden_states = outputs.last_hidden_state

# 各単語の埋め込みベクトルを取得
word_embeddings = []

# トークンごとに埋め込みベクトルを取得
for i, token in enumerate(tokens["input_ids"][0]):
    # トークンの位置に対応する埋め込みベクトルを抽出
    embedding_vector = last_hidden_states[0, i].detach().numpy()
    word_embeddings.append((tokenizer.convert_ids_to_tokens(token.item()), embedding_vector))

# 全ての単語の埋め込みベクトルを取り出す
all_embeddings = [embedding for _, embedding in word_embeddings]

# リストをNumPy配列に変換
all_embeddings = np.array(all_embeddings)

# t-SNEを使用して2次元に変換
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
transformed_vectors = tsne.fit_transform(all_embeddings)

# 変換後のベクトルをプロット
plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1])

# 各点に単語のラベルを追加
for i, (word, _) in enumerate(word_embeddings):
    plt.annotate(word, (transformed_vectors[i, 0], transformed_vectors[i, 1]))

plt.title("t-SNE Visualization of Word Vectors")
plt.show()