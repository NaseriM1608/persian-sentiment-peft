import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.config import TrainingConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data.dataset import SentimentDataset
from peft import PeftModel, PeftConfig
from src.training.train import compute_metrics


def evaluate():
    config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    test_dataset = SentimentDataset(config.test_path, tokenizer)

    peft_config = PeftConfig.from_pretrained(config.output_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path,
                                                                    num_labels=config.num_labels)
    model = PeftModel.from_pretrained(base_model, config.output_dir)

    training_args = TrainingArguments(
        output_dir="checkpoints/"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(test_dataset)
    print(results)


if __name__ == "__main__":
    evaluate()

