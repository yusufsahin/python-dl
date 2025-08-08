import torch
import torch.nn as nn
import torch.optim as optim


# 🔤 Shakespeare'den daha uzun bir metin (manuel alınmış)
text = (
    "To be, or not to be: that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die to sleep;\n"
    "From fairest creatures we desire increase\n"
    "That thereby beauty’s rose might never die\n"
    "But as the riper should by time decease\n"
    "His tender heir might bear his memory:\n"
    "But thou contracted to thine own bright eyes\n"
    "Feed’st thy light’s flame with self-substantial fuel\n"
    "Making a famine where abundance lies\n"
    "Thyself thy foe, to thy sweet self too cruel:\n"
    "Thou that art now the world’s fresh ornament\n"
    "And only herald to the gaudy spring\n"
    "Within thine own bud buriest thy content\n"
    "And, tender churl, mak’st waste in niggarding:\n"   
    "Pity the world, or else this glutton be\n"
    "To eat the world’s due, by the grave and thee.\n"
)

# 🔠 Karakter eşlemeleri
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
seq_length = 60
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧩 Dataset oluşturma
def create_dataset(text, seq_length):
    X, Y = [], []
    for i in range(len(text) - seq_length):
        X.append([char_to_idx[c] for c in text[i:i+seq_length]])
        Y.append([char_to_idx[c] for c in text[i+1:i+seq_length+1]])
    return torch.tensor(X), torch.tensor(Y)

X, Y = create_dataset(text, seq_length)
X, Y = X.to(device), Y.to(device)

# 🧠 RNN modeli
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

model = CharRNN(vocab_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# 🏋️‍♂️ Eğitim
for epoch in range(200):
    model.train()
    hidden = torch.zeros(1, X.size(0), hidden_size).to(device)
    output, _ = model(X, hidden)
    loss = criterion(output.view(-1, vocab_size), Y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ✨ Metin üretme
def generate_text(start_seq, length=300):
    model.eval()
    input_seq = torch.tensor([char_to_idx[c] for c in start_seq], dtype=torch.long).unsqueeze(0).to(device)
    hidden = torch.zeros(1, 1, hidden_size).to(device)
    result = start_seq

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[:, -1, :], dim=-1).squeeze()
            char_idx = torch.multinomial(probs, 1).item()
            result += idx_to_char[char_idx]
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[char_idx]]).to(device)], dim=1)

    return result


# 🚀 Örnek çıktı
print("\n📝 Üretilen Shakespeare tarzı metin:\n")
print(generate_text("To be, or not to be", 400))
