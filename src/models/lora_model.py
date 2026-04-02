from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased", num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )

    peft_model =  get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model