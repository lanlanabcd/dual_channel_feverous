# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/13 20:17
# Description:
# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/28 16:30
# Description:

from torch.utils.data.dataset import Dataset
import torch
from my_utils import load_jsonl_data, refine_obj_data
from tqdm import tqdm

def collate_fn(batch):
    raw_data, inputs, label = map(list, zip(*batch))
    batched_raw_data = raw_data
    batch_inputs = torch.stack(inputs)
    batched_label = torch.stack(label)

    return batched_raw_data, batch_inputs, batched_label


class BaseGenerator(Dataset):
    def __init__(self, input_path, data_type, args):
        super(BaseGenerator, self).__init__()
        self.model_name = str(type(self))
        self.args = args
        self.config = args.config
        self.data_type = data_type

        self.raw_data = self.preprocess_raw_data(self.get_raw_data(input_path, keys=self.get_refine_keys()))
        self.labels = self.get_labels(self.raw_data)
        assert len(self.labels) != 0

    def get_labels(self, data):
        labels = [self.config.label2idx.get(entry["label"], 0) for entry in data]
        return labels

    def print_example(self):
        instance = self.raw_data[0]
        for k, v in instance.items():
            print(k, " : ", v)

        instance = self.raw_data[-1]
        for k, v in instance.items():
            print(k, " : ", v)

    def preprocess_raw_data(self, raw_data):
        return raw_data

    def get_refine_keys(self):
        keys = None
        return keys

    def get_raw_data(self, input_path, keys=None):
        raw_data = load_jsonl_data(input_path)
        if keys is not None:
            raw_data = refine_obj_data(raw_data, keys)
        return raw_data

    @classmethod
    def collate_fn(cls):
        return collate_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        inputs = raw_data["inputs"]
        label = torch.tensor(self.labels[idx]).to(self.args.device)
        return raw_data, inputs, label


class MyArgs():
    def __init__(self):
        self.device = "cpu"
        self.label2idx = {
            "FALSE": 0,
            "TRUE": 1
        }
        self.config = {}


if __name__ == "__main__":
    input_path = "../data/mla_subevis_data/dev.jsonl"

    args = MyArgs()
    generator = BaseGenerator(input_path, "dev", args)

    data_len = generator.__len__()
    print(generator.__getitem__(0))
    print(data_len)
    for i in tqdm(range(data_len)):
        generator.__getitem__(i)
