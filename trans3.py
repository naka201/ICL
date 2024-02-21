from transformers import BertTokenizer, BertModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

# モデルとトークナイザのロード
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")

# 複数の日本語単語(データ数100)
words = ['日本', '東京', 'スペイン', 'マドリード',
         'フランス', 'パリ', 'イタリア', 'ローマ', 
         'ドイツ', 'ベルリン', 'イギリス', 'ロンドン', 
         'ギリシャ', 'アテネ', 'オランダ', 'アムステルダム', 
         'フィンランド', 'ヘルシンキ', 'ポーランド', 'ワルシャワ',
         'ノルウェー', 'オスロ', 'スウェーデン', 'ストックホルム', 
         'ベルギー', 'ブリュッセル', 'オーストリア', 'ウィーン', 
         'チェコ', 'プラハ', 'デンマーク', 'コペンハーゲン', 
         'ルワンダ', 'キガリ', 'イスラエル', 'エルサレム', 
         'ベトナム', 'ハノイ', 'サウジアラビア', 'リヤド', 
         'タイ', 'バンコク', 'ジンバブエ', 'ハラレ', 
         'フィリピン', 'マニラ', 'レバノン', 'ベイルート', 
         'インドネシア', 'ジャカルタ', 'カタール', 'ドーハ', 
         'バングラデシュ', 'ダッカ', 'シリア', 'ダマスカス', 
         'イラク', 'バグダッド', 'カンボジア', 'プノンペン', 
         'ウズベキスタン', 'タシケント', 'モンゴル', 'ウランバートル', 
         'ブラジル', 'ブラジリア', 'ペルー', 'リマ', 
         'アルゼンチン', 'ブエノスアイレス', 'コロンビア', 'ボゴタ', 
         'エクアドル', 'キト', 'ベネズエラ', 'カラカス', 
         'チリ', 'サンティアゴ', 'ウルグアイ', 'モンテビデオ', 
         'パラグアイ', 'アスンシオン', 'モロッコ', 'ラバト', 
         'リビア', 'トリポリ', 'アルジェリア', 'アルジェ', 
         'ナイジェリア', 'アブジャ', 'チュニジア', 'チュニス', 
         'ケニア', 'ナイロビ', 'ガーナ', 'アクラ', 
         'チャド', 'ンジャメナ', 'カメルーン', 'ヤウンデ']


# 各単語をトークン化
tokenized_words = tokenizer(words, return_tensors="pt", padding=True, truncation=True)

# モデルにトークンを入力して出力を取得
outputs = model(**tokenized_words)

# 最終層の隠れ状態を取得
last_hidden_states = outputs.last_hidden_state

# 各単語の埋め込みベクトルを抽出
word_embeddings = [last_hidden_states[i, 0].detach().numpy() for i in range(len(words))]

# t-SNEで2次元に変換
tsne_model = TSNE(n_components=2, perplexity=50,random_state=42)
tsne_embeddings = tsne_model.fit_transform(np.array(word_embeddings))

# 2次元に変換されたベクトルの表示
for word, vector in zip(words, tsne_embeddings):
    print(f'{word}: {vector}')

# プロット
plt.figure(figsize=(8, 8))
for word, vector in zip(words, tsne_embeddings):
    plt.scatter(vector[0], vector[1])
    plt.annotate(word, (vector[0], vector[1]), alpha=0.7)

plt.title('t-SNE Visualization of Japanese Word Embeddings')
plt.show()