import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.configuration.config import *


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts: str | list):
        is_str = isinstance(texts, str)
        if is_str:
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        labels = [self.model.config.id2label[pred_id] for pred_id in preds]

        if is_str:
            return labels[0]

        return labels


def validate_model_tokenizer(model, tokenizer):
    model_vocab_size = getattr(model.config, "vocab_size", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if model_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "Model and tokenizer vocab sizes do not match. "
            "Regenerate processed data and retrain the model after changing BERT_MODEL_NAME."
        )


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Path(MODEL_DIR) / "best")
    validate_model_tokenizer(model, tokenizer)
    predictor = Predictor(model, tokenizer, device)

    text = "gold wave"
    result = predictor.predict(text)
    print(f"Input text: {text}")
    print(f"Predicted label: {result}")

    texts = ["gold wave", "no huddle gold", "gold sparkle", "white tiger"]
    results = predictor.predict(texts)
    for text, label in zip(texts, results):
        print(f"Input text: {text}")
        print(f"Predicted label: {label}")


if __name__ == "__main__":
    predict()
