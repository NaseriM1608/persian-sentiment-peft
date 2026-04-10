from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from src.training.config import TrainingConfig
import torch

config = TrainingConfig()

peft_config = PeftConfig.from_pretrained(config.output_dir)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=config.num_labels
)
model = PeftModel.from_pretrained(base_model, config.output_dir)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()
    label = "HAPPY" if predicted_class == 0 else "SAD"
    return {"label": label, "label_id": predicted_class}