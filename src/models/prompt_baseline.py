from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()


def zero_shot_predict(text):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()


def few_shot_predict(text):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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
            {"role": "user", "content": "همیشه زودتر از موعد میرسه واقعا خسته نباشید میگم بهتون"},
            {"role": "assistant", "content": "HAPPY"},
            {"role": "user", "content": text},

        ],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()

