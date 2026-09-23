from dataclasses import dataclass
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.configuration.config import *
from src.process.dataset import get_dataloader, get_dataset


@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    save_steps: int = SAVE_STEPS
    output_dir: str = str(MODEL_DIR)
    log_dir: str = str(LOG_DIR)
    early_stop_metric: str = "loss"
    early_stop_patience: int = 2
    use_amp: bool = True


class Trainer:
    def __init__(self, model, valid_dataset, collate_fn, device, train_dataset=None, train_config=TrainConfig()):
        self.train_config = train_config
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)
        self.step = 1
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d-%H-%M-%S")))
        self.early_stop_best_score = -float("inf")
        self.early_stop_counter = 0

        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            init_scale=2.0**16,
            enabled=self.train_config.use_amp,
        )

        self.checkpoint_path = Path(self.train_config.output_dir) / "last" / "checkpoint.pt"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_dataloader(self, dataset):
        dataset.set_format(type="torch")
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def train(self):
        self._load_checkpoint()
        dataloader = self._get_dataloader(self.train_dataset)

        for epoch in range(self.train_config.epochs):
            for inputs in tqdm(dataloader, desc=f"[Epoch: {epoch + 1}]"):
                this_loss = self._train_one_step(inputs)
                self.writer.add_scalar("train/loss", this_loss, self.step)

                if self.step % self.train_config.save_steps == 0:
                    tqdm.write(f"[Epoch:{epoch + 1} | Step:{self.step}] Loss: {this_loss}")
                    self.writer.add_scalar("loss", this_loss, self.step)

                    metrics = self.evaluate()
                    metrics_str = "|".join([f"{key}:{value:.4f}" for key, value in metrics.items()])
                    tqdm.write(f"[Evaluate: {metrics_str}]")

                    if self._should_stop(metrics):
                        tqdm.write("[early stop]")
                        return

                    self._save_checkpoint()

                self.step += 1

    def _train_one_step(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.train_config.use_amp,
        ):
            outputs = self.model(**inputs)
            loss = outputs.loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self):
        dataloader = self._get_dataloader(self.valid_dataset)
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs in tqdm(dataloader, desc="[Evaluate]"):
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(inputs["labels"].tolist())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        self.model.train()
        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    def _should_stop(self, metrics):
        metric = metrics[self.train_config.early_stop_metric]
        score = -metric if self.train_config.early_stop_metric == "loss" else metric

        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_counter = 0
            tqdm.write("Saving best model...")
            self.model.save_pretrained(str(Path(self.train_config.output_dir) / "best"))
            return False

        self.early_stop_counter += 1
        return self.early_stop_counter >= self.train_config.early_stop_patience

    def _save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "step": self.step,
            "early_stop_best_score": self.early_stop_best_score,
            "early_stop_counter": self.early_stop_counter,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.step = checkpoint["step"]
            self.early_stop_best_score = checkpoint["early_stop_best_score"]
            self.early_stop_counter = checkpoint["early_stop_counter"]
            print(f"Checkpoint loaded at step {self.step}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    with open(MODEL_DIR / LABELS_FILE, "r", encoding="utf-8") as f:
        all_labels = f.read().splitlines()
        num_labels = len(all_labels)

    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print(model.config.id2label)
    model.save_pretrained(MODEL_DIR)

    train_dataset = get_dataset("train").select(range(1000))
    valid_dataset = get_dataset("valid").select(range(100))
    get_dataloader(tokenizer)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        device=device,
        train_config=TrainConfig(),
    )

    trainer.train()


if __name__ == "__main__":
    train()
