import tiktoken
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
            self.input_ids.append(self.tokens[i : i + max_length])
            self.target_ids.append(self.tokens[i + 1 : i + max_length + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(text, mode_name, max_length, stride, batch_size, shuffle):
    dataset = BPETokenizer(text, mode_name, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    dataloader = iter(
        create_dataloader(
            open("the-verdict.txt", "r").read(), "gpt-3.5-turbo", 4, 1, 1, False
        )
    )
    print(next(dataloader))
    print(next(dataloader))
