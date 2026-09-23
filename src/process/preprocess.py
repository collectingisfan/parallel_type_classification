from datasets import load_dataset
from transformers import AutoTokenizer

from src.configuration.config import *


def preprocess():
    dataset_dict = load_dataset(
        "csv",
        data_files={
            "train": str(RAW_DATA_DIR / RAW_TRAIN_DATA),
            "valid": str(RAW_DATA_DIR / RAW_VALID_DATA),
            "test": str(RAW_DATA_DIR / RAW_TEST_DATA),
        },
        delimiter="\t",
    )

    def copy_label(example):
        example["label_text"] = example["label"]
        return example

    dataset_dict = dataset_dict.map(copy_label)
    dataset_dict = dataset_dict.class_encode_column("label")

    label_feature = dataset_dict["train"].features["label"]
    print("Labels:", label_feature.names)

    with open(MODEL_DIR / LABELS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(label_feature.names))

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def batch_encode(examples):
        inputs = tokenizer(examples["text_a"], truncation=True)
        inputs["labels"] = examples["label"]
        return inputs

    dataset_dict = dataset_dict.map(
        batch_encode,
        batched=True,
        remove_columns=["label", "text_a", "label_text"],
    )
    print(dataset_dict["train"][0:3])

    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == "__main__":
    preprocess()
