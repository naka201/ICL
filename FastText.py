import fasttext
import numpy as np

model_path = "/Users/nakanishitakumi/Downloads/cc.en.300.bin"
model = fasttext.load_model(model_path)

# 単語の取得
with open('data/country.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

embeddings = [model[word] for word in words]

# NumPyの配列に変換
embeddings = np.array(embeddings)

word_size = int(len(words)/2)

def comp(embeddings):
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    for i in range(0, 10, 2):
        val1 = embeddings[i+1] - embeddings[i]
        results = []
        # Calculate for each j
        for j in range(0, 10, 2):
            if j != i:
                val2 = val1 + embeddings[j]
                results_for_j = []

                for k in range(10):
                    vec_cal = (cos_sim(val2, embeddings[k]))
                    results_for_j.append((f"{words[k]}", vec_cal))

                sorted_results_for_j = sorted(results_for_j, key=lambda x: x[1], reverse=True)[:5]
                print(f"Results for {words[i+1]} - {words[i]} + {words[j]}:")
                for idx, (word_k, vec_cal) in enumerate(sorted_results_for_j, start=1):
                    print(f"  {idx}. {word_k}: {format(vec_cal, '.4f')}")
                    if idx == 1 and words.index(word_k) == j + 1:
                        print("  Good!")
                results.extend(results_for_j)
    
def cos_cal(embeddings):
    print("コサイン類似度計算")
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

def main():
    comp(embeddings)
    #cos_cal(embeddings)
    #tsne(embeddings)
    #pca(embeddings)

if __name__ == "__main__":
    main()