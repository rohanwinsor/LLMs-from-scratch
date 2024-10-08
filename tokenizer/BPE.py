import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class BPETokenizer(Dataset):
    def __init__(
        self, text, model_name="gpt-3.5-turbo", max_length=4, stride=1
    ) -> None:
        super().__init__()
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.tokens = self.tokenizer.encode(text)
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(self.tokens) - max_length, stride):
            self.input_ids.append(torch.tensor(self.tokens[i : i + max_length]))
            self.target_ids.append(
                torch.tensor(self.tokens[i + 1 : i + max_length + 1])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def add_pos_embeddings(ids, n_vocab=50257, n_dim=256):
    embedding_layer = nn.Embedding(n_vocab, n_dim) # token ids 
    pos_embedding_layer = nn.Embedding(n_vocab, n_dim) # pos encoding
    return embedding_layer(ids) + pos_embedding_layer(torch.arange(len(ids))) # len(ids) == max_len


def create_dataloader(text, mode_name, max_length, stride, batch_size, shuffle):
    dataset = BPETokenizer(text, mode_name, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    dataloader = iter(
        create_dataloader(open("the-verdict.txt", "r").read(), "gpt2", 4, 1, 1, True)
    )
    X, y = next(dataloader)
    print(type(X))
    print(add_pos_embeddings(X))
    print(add_pos_embeddings(y))
