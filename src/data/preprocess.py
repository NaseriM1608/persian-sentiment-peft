from hazm import *
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data():
    df = pd.read_csv("data/raw/Snappfood - Sentiment Analysis.csv", sep="\t", on_bad_lines="skip")
    df = df.dropna(subset=["label_id"])

    normalizer = Normalizer()

    texts = list(df.comment)
    labels = [int(label) for label in df.label_id]

    cleaned_texts = [normalizer.normalize(text) for text in texts]

    X_train, X_temp, y_train, y_temp = train_test_split(
        cleaned_texts, labels, test_size=0.3, random_state=10
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=10
    )

    train_df = pd.DataFrame({
        "text": X_train,
        "label_id": y_train
    })

    val_df = pd.DataFrame({
        "text": X_val,
        "label_id": y_val
    })

    test_df = pd.DataFrame({
        "text": X_test,
        "label_id": y_test
    })

    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)

if __name__ == "__main__":
    prepare_data()