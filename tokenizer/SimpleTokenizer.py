import re
from typing import Set


class SimpleTokenizerV2:
    def __init__(self, vocab: Set[str]):
        self.vocab: Set[str] = vocab
        self.vocab.add("<unk>")
        self.vocab.add("<|endoftext|>")
        self.idx2c = {idx: c for idx, c in enumerate(self.vocab)} # in the book this is vocab
        self.c2idx = {c: idx for idx, c in self.idx2c.items()}

    @staticmethod
    def reg_exp():
        return r'([,.:;?_!"()\']|--|\s)'
    def encode(self, text):
        tokens = [i for i in re.split(SimpleTokenizerV2.reg_exp(), text) if i.strip()]
        return [self.c2idx.get(tok, self.c2idx["<unk>"]) for tok in tokens]

    def decode(self, tokens):
        text = ' '.join(self.idx2c[tok] for tok in tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # WHY !!!!
        return text
    
if __name__ == "__main__":
    with open("the-verdict.txt", "r") as f:
        data = f.read()
    vocab = set([i for i in re.split(SimpleTokenizerV2.reg_exp(), data) if i.strip()])
    tokenzier = SimpleTokenizerV2(vocab)
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    print(tokenzier.decode(tokenzier.encode(text)))