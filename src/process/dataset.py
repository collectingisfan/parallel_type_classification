from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.configuration.config import *


def get_dataset(ds_type):
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)
    return dataset


def get_dataloader(tokenizer, ds_type="train"):
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)
    dataset.set_format(type="torch")

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    dataloader = get_dataloader(tokenizer)
    for batch in dataloader:
        for key, value in batch.items():
            print(key, value)
        break
