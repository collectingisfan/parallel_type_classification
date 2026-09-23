import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.configuration.config import *
from src.process.dataset import get_dataset
from src.runner.predict import validate_model_tokenizer
from src.runner.train import Trainer


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "best")
    validate_model_tokenizer(model, tokenizer)

    test_dataset = get_dataset("test")
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        valid_dataset=test_dataset,
        collate_fn=collate_fn,
        device=device,
    )

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    evaluate()
