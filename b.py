import openai
import numpy as np
import torch

# APIキーを設定
api_key = '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'
openai.api_key = api_key

# 単語の取得
with open('data/country.txt', 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file]

embeddings = []
for word in words:
    embeddings.append(openai.Embedding.create(model="text-embedding-ada-002", input=word).data[0].embedding)

embeddings = np.array(embeddings)
v1 = embeddings[0] - embeddings[1]
v2 = embeddings[2] - embeddings[3]

embedding0 = torch.tensor(v1)
embedding1 = torch.tensor(v2)

# cosine_similarityを計算
result = torch.nn.functional.cosine_similarity(embedding0, embedding1, dim=0)
print(result.item())