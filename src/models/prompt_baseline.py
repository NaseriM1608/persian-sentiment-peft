from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

load_dotenv()

client = Groq()


def zero_shot_predict(text):
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content":
                """
                    You are a sentiment analysis assistant specialized in Persian (Farsi) text.
    
                    Your task is to analyze customer reviews written in Persian and classify their sentiment.
                    
                    Follow these rules strictly:
                    - The input will always be a Persian customer review.
                    - You must determine whether the sentiment of the review is positive or negative.
                    - If the sentiment is positive, output: HAPPY
                    - If the sentiment is negative, output: SAD
                    - Do not output anything else under any circumstances.
                    - Do not explain your reasoning.
                    - Do not add punctuation, extra words, or formatting.
                    - Your entire response must be exactly one word: HAPPY or SAD
                    
                    Guidelines for classification:
                    - HAPPY → indicates satisfaction, praise, approval, or a good experience
                    - SAD → indicates dissatisfaction, complaints, criticism, or a bad experience
                    - If the sentiment is mixed, choose the dominant sentiment
                    - If uncertain, choose the closest overall sentiment based on tone
                    
                    Now classify the following Persian review:
                """ + f"{text}"
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()


def few_shot_predict(text):
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content":
                    """
                        You are a sentiment analysis assistant specialized in Persian (Farsi) text.

                        Your task is to analyze customer reviews written in Persian and classify their sentiment.

                        Follow these rules strictly:
                        - The input will always be a Persian customer review.
                        - You must determine whether the sentiment of the review is positive or negative.
                        - If the sentiment is positive, output: HAPPY
                        - If the sentiment is negative, output: SAD
                        - Do not output anything else under any circumstances.
                        - Do not explain your reasoning.
                        - Do not add punctuation, extra words, or formatting.
                        - Your entire response must be exactly one word: HAPPY or SAD

                        Guidelines for classification:
                        - HAPPY → indicates satisfaction, praise, approval, or a good experience
                        - SAD → indicates dissatisfaction, complaints, criticism, or a bad experience
                        - If the sentiment is mixed, choose the dominant sentiment
                        - If uncertain, choose the closest overall sentiment based on tone

                        Now classify the following Persian review:
                    """
            },
            {"role": "user", "content": "سرویس دهیتون واقعا عالی بود. قراره همیشه از اینجا خرید کنم"},
            {"role": "assistant", "content": "HAPPY"},
            {"role": "user", "content": "این چه وضعشه خدا وکیلی غذا رو نه تنها دیر اوردید بلکه سرد هم بود"},
            {"role": "assistant", "content": "SAD"},
            {"role": "user", "content": text},

        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()

def evaluate_baseline():
    df = pd.read_csv("data/splits/test.csv").sample(200, random_state=42)
    texts = list(df.text)
    true_labels = list(df.label_id)

    label_map = {"HAPPY": 0, "SAD": 1}

    zero_shot_preds = []
    few_shot_preds = []

    for text in texts:
        zero_shot_preds.append(label_map.get(zero_shot_predict(text), 0))
        few_shot_preds.append(label_map.get(few_shot_predict(text), 0))

    print("Zero-shot:")
    print(f"  F1: {f1_score(true_labels, zero_shot_preds, average='weighted'):.4f}")
    print(f"  Accuracy: {accuracy_score(true_labels, zero_shot_preds):.4f}")

    print("Few-shot:")
    print(f"  F1: {f1_score(true_labels, few_shot_preds, average='weighted'):.4f}")
    print(f"  Accuracy: {accuracy_score(true_labels, few_shot_preds):.4f}")


if __name__ == "__main__":
    evaluate_baseline()

