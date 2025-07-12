import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset import TextDataset
from src.model import LSTMClassifier
from src.train import train_model
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# 1. Load data
df = pd.read_csv("data/IMDB Dataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 2. Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# 3. Tokenizer basic
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

# 4. Dataset
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), vocab, tokenizer)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), vocab, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 5. Model
model = LSTMClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 6. Train
train_model(model, train_loader, val_loader, device, lr=1e-3, epochs=5)

# 7. Save
torch.save(model.state_dict(), "models/sentiment_model.pt")
