from transformers import AutoTokenizer, OpenAIGPTModel
import torch
import numpy as np

# GPT-2のトークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = OpenAIGPTModel.from_pretrained("openai-gpt")

# 単語の取得
with open('data/country.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]
    
# 各単語のトークンIDを取得
tokenized_inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)

# トークンIDからembedding matrixを用いて埋め込みを取得
vocab_size = tokenizer.vocab_size
embeddings = model.get_input_embeddings()(tokenized_inputs["input_ids"]).detach().numpy()[:, 0, :]

embeddings = np.array(embeddings)

def cos(embeddings):
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    va1 = embeddings[1] - embeddings[0]
    for i in range(2,10,2):
        v2 = va1 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[1]} - {words[0]} + {words[i]} = {words[i+1]}? : {vec_cal}")
    
    va2 = embeddings[3] - embeddings[2]
    v2 = va2 + embeddings[0]
    print(f"{words[3]} - {words[2]} + {words[1]} = {words[0]}? : {vec_cal}")
    for i in range(4,10,2):
        v2 = va2 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[3]} - {words[2]} + {words[i]} = {words[i+1]}? : {vec_cal}")

    va3 = embeddings[5] - embeddings[4]
    for i in range(0,4,2):
        v2 = va3 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[5]} - {words[4]} + {words[i]} = {words[i+1]}? : {vec_cal}")
    for i in range(6,10,2):
        v2 = va3 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[5]} - {words[4]} + {words[i]} = {words[i+1]}? : {vec_cal}")
    
    va4 = embeddings[7] - embeddings[6]
    for i in range(0,6,2):
        v2 = va4 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[7]} - {words[6]} + {words[i]} = {words[i+1]}? : {vec_cal}")
    for i in range(8,10,2):
        v2 = va4 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[7]} - {words[6]} + {words[i]} = {words[i+1]}? : {vec_cal}")

    va5 = embeddings[9] - embeddings[8]
    for i in range(0,8,2):
        v2 = va5 + embeddings[i]
        vec_cal = format(cos_sim(v2, embeddings[i+1]), '.2f')
        print(f"{words[9]} - {words[8]} + {words[i]} = {words[i+1]}? : {vec_cal}")


def cos_cal(embeddings):
    print("コサイン類似度計算")
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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib

def tsne(embeddings):
    embeddings = np.array(embeddings)
    # t-SNEで2次元に変換
    tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_embeddings = tsne_model.fit_transform(embeddings)

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

#PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca(embeddings):
    # PCAで2次元に変換
    pca_model = PCA(n_components=2)
    pca_embeddings = pca_model.fit_transform(embeddings)

    # プロット
    plt.figure(figsize=(8, 8))
    for word, vector in zip(words, pca_embeddings):
        plt.scatter(vector[0], vector[1])
        plt.annotate(word, (vector[0], vector[1]), alpha=0.7)

    # x軸に線を引く
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    # y軸に線を引く
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)

    plt.title('PCA Visualization of Word Embeddings')
    plt.show()


def main():
    cos(embeddings)
    cos_cal(embeddings)
    #tsne(embeddings)
    #pca(embeddings)

if __name__ == "__main__":
    main()

