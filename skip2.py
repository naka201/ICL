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

# 複数の単語
words = ['日本', 'スペイン', 'フランス', 'イタリア',
         'ドイツ', 'イギリス', 'ギリシャ', 'オランダ', 
         'フィンランド', 'ポーランド', 'ノルウェー', 'スウェーデン', 
         'ベルギー', 'オーストリア', 'チェコ',
         '東京', 'マドリード', 'パリ', 'ローマ', 
         'ベルリン', 'ロンドン', 'アテネ', 'アムステルダム', 
         'ヘルシンキ', 'ワルシャワ', 'オスロ', 'ストックホルム', 
         'ブリュッセル','ウィーン', 'プラハ']

# 複数の単語の埋め込みベクトルを取得
embedding_vectors = [embedding_layer[word] for word in words]

# NumPyの配列に変換
embedding_vectors = np.array(embedding_vectors)

# t-SNEで2次元に変換
tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_embeddings = tsne_model.fit_transform(embedding_vectors)

'''
# 2次元に変換されたベクトルの表示
for word, vector in zip(words_to_embed, tsne_embeddings):
    print(f'{word}: {vector}')
'''
    
num_words = len(words)

for i in range(num_words // 2):
    point1 = tsne_embeddings[i]
    point2 = tsne_embeddings[i + 1]

    a = (point1[1] - point2[1])/(point1[0] - point2[0])
    print(f'{words[i]}と{words[i+num_words//2]}の傾き: {a}')


# プロット
plt.figure(figsize=(8, 8))
for word, vector in zip(words, tsne_embeddings):
    plt.scatter(vector[0], vector[1])
    plt.annotate(word, (vector[0], vector[1]), alpha=0.7)

plt.title('t-SNE Visualization of Word Embeddings')
plt.show()
