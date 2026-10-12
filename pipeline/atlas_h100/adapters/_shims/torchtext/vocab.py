"""Minimal torchtext.vocab: Vocab with the methods scGPT's GeneVocab calls, and vocab()."""
from __future__ import annotations
from collections import OrderedDict


class _Store:
    def __init__(self, itos=None):
        self.itos = list(itos or [])
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.default_index = None


class Vocab:
    def __init__(self, vocab=None):
        self.vocab = vocab if isinstance(vocab, _Store) else _Store(vocab)

    def __len__(self):
        return len(self.vocab.itos)

    def __contains__(self, token):
        return token in self.vocab.stoi

    def __getitem__(self, token):
        i = self.vocab.stoi.get(token)
        if i is None:
            if self.vocab.default_index is None:
                raise RuntimeError(f"Token {token} not found and default index is not set")
            return self.vocab.default_index
        return i

    def __call__(self, tokens):
        return self.lookup_indices(tokens)

    def insert_token(self, token, index):
        if token in self.vocab.stoi:
            raise RuntimeError(f"Token {token} already exists in the Vocab")
        if not 0 <= index <= len(self.vocab.itos):
            raise RuntimeError(f"Index {index} out of range")
        self.vocab.itos.insert(index, token)
        self.vocab.stoi = {t: i for i, t in enumerate(self.vocab.itos)}

    def append_token(self, token):
        self.insert_token(token, len(self.vocab.itos))

    def set_default_index(self, index):
        self.vocab.default_index = index

    def get_default_index(self):
        return self.vocab.default_index

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

    def lookup_token(self, index):
        return self.vocab.itos[index]

    def lookup_tokens(self, indices):
        return [self.vocab.itos[i] for i in indices]

    def get_stoi(self):
        return dict(self.vocab.stoi)

    def get_itos(self):
        return list(self.vocab.itos)


def vocab(ordered_dict, min_freq=1, specials=None, special_first=True):
    tokens = [t for t, f in OrderedDict(ordered_dict).items() if f >= min_freq]
    specials = [s for s in (specials or []) if s not in tokens]
    return Vocab(_Store(specials + tokens if special_first else tokens + specials))
