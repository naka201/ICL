from transformers import BertModel, BertTokenizer
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 事前訓練済みのBERTモデルとトークナイザを取得
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# テキストをトークナイズ
text = "Tokyo"
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

# 各単語とその埋め込みベクトルを表示
for word, embedding in word_embeddings:
    print(f"{word}: {embedding.shape}")


