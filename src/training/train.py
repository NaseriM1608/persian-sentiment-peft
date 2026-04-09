from config import TrainingConfig
from src.data.dataset import SentimentDataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from src.models.lora_model import load_model
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {
        "f1": f1,
        "accuracy": accuracy
    }


def train():
    config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = SentimentDataset(csv_path=config.train_path, tokenizer=tokenizer)
    val_dataset = SentimentDataset(csv_path=config.val_path, tokenizer=tokenizer)

    lora_model = load_model()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="../../experiments/",
        logging_steps=50,
        fp16=True,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()