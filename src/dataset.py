import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        token_ids = self.vocab(tokens)[:self.max_len]
        padding = [self.vocab["<pad>"]] * (self.max_len - len(token_ids))
        input_ids = token_ids + padding
        return torch.tensor(input_ids), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.texts)
