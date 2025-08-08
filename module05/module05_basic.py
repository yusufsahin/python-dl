import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

texts = ['dog bites man', 'man bites dog', 'dog eats meat', 'man eats food']

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# One-hot vektör gösterimi (sparse)
one_hot_matrix = tokenizer.texts_to_matrix(texts, mode='binary')
# Embedding layer ile dense temsil
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=8))
model.compile('rmsprop', 'mse')
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=4)
embedding_output = model.predict(padded)

# Ortalama vektör (sentence representation) alınır
sentence_embeddings = np.mean(embedding_output, axis=1)


# t-SNE ile görselleştir (🛠️ FIX: perplexity < n_samples)
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
reduced = tsne.fit_transform(sentence_embeddings)


# Görselleştirme
plt.figure(figsize=(8, 5))
for i, label in enumerate(texts):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
plt.title("One-Hot vs. Embedding Temsili (t-SNE)")
plt.grid()
plt.show()


