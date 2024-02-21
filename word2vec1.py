import gensim.downloader as api

word_vectors = api.load("word2vec-google-news-300")

result = word_vectors.most_similar(positive=['brother', 'sister'], negative=['father'])

# 類似度の高い順にソート
sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

# 上位5つを表示
for i, (word, similarity) in enumerate(sorted_result[:5], 1):
    print(f"{i}. {word}: {similarity:.4f}")
