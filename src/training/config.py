from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "HooshvareLab/bert-fa-base-uncased"
    train_path: str = "../../data/splits/train.csv"
    val_path: str = "../../data/splits/val.csv"
    test_path: str = "../../data/splits/test.csv"
    num_labels: int = 2
    max_length:int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-4
    output_dir: str = "../../checkpoints/"
