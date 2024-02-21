import torch
from trans5 import TransformerModel, PositionalEncoding, generate_square_subsequent_mask

# Load the saved model
loaded_model = torch.load('transformer_model.pth')

# Access the embedding layer of the loaded model
embedding_layer = loaded_model.encoder

# Define vocab and device
train_iter, _, _ = WikiText2()
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get indices for words
word_indices = [vocab[word] for word in ['Japan', 'Tokyo', 'Hello']]

# Convert to Tensor
word_indices_tensor = torch.tensor(word_indices, dtype=torch.long).to(device)

# Use the embedding layer to get embeddings for words
embeddings = embedding_layer(word_indices_tensor)

# Convert to NumPy array if needed
embeddings_numpy = embeddings.cpu().detach().numpy()

# Check the result
print(embeddings_numpy)
