import torch
from torch.utils.data import Dataset
import random
import logging
import json

class NERDataset(Dataset):
    def __init__(self, file, hp=None):
        super(NERDataset, self).__init__()
        if isinstance(file, list):
            self.data = file
        else:
            self.data = self._load_data(file)
        self.corpus_num = len(self.data)
        self.hp = hp

    def _load_data(self, file):
        data = []
        for line in open(file, "r", encoding="utf-8"):
            line_data = json.loads(line)
            ids = line_data["input_ids"]
            labels = line_data["labels"]
            sample = {"ids": ids, "labels": labels}
            data.append(sample)
        random.shuffle(data)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


    def collate_wrapper(self, batch):
        max_seq_len = max([len(s["ids"]) for s in batch])
        input_ids_labels = [[s['ids'], s['labels']] for s in batch]
        input_ids_labels = sorted(input_ids_labels, key=lambda x: len(x[0]), reverse=True)
        seqs_length = [len(u[0]) for u in input_ids_labels]
        input_ids = [u[0] + [0] * (max_seq_len - len(u[0])) for u in input_ids_labels]
        labels = [u[1] + [0] * (max_seq_len - len(u[1])) for u in input_ids_labels]

        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        attention_mask = torch.as_tensor(input_ids.ne(0), dtype=torch.long)

        samples = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
                   "seqs_length": seqs_length}

        return samples
