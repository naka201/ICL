'''
import cohere

api_key = 'uopo8n5C7Bbg7BQeYxv1cSUVlDWdItUsvLKTWsdw'

co = cohere.Client(api_key)
co.check_api_key()

words_to_embed = ['hello', 'world', 'python', 'cohere']
res_embed = co.embed(
    texts=words_to_embed,
    model='embed-english-light-v2.0'
)

# 埋め込みベクトルを表示
for word, embedding in zip(words_to_embed, res_embed['embeddings']):
    print(f'Word: {word}, Embedding: {embedding}')
'''

import cohere
import numpy as np

co = cohere.Client("uopo8n5C7Bbg7BQeYxv1cSUVlDWdItUsvLKTWsdw")

# get the embeddings
phrases = ["japan", "tokyo", "apple"]
(soup1, soup2, london) = co.embed(phrases).embeddings

# compare them
def calculate_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

calculate_similarity(soup1, soup2) # 0.9 - very similar!
calculate_similarity(soup1, london) # 0.3 - not similar!