# Persian Sentiment Analysis — LoRA Fine-tuning vs Prompting

A benchmark study comparing LoRA fine-tuning of a Persian-native BERT model against zero-shot and few-shot prompting for sentiment analysis on Persian customer reviews.

## Research Question

> Is LoRA fine-tuning worth it for Persian sentiment analysis, or can prompting a general-purpose LLM achieve comparable results?

## Results

| Approach | Model | F1 | Accuracy |
|---|---|---|---|
| LoRA Fine-tuning | ParsBERT (HooshvareLab/bert-fa-base-uncased) | 0.8713 | 0.8714 |
| Few-shot Prompting | meta-llama/llama-4-scout-17b-16e-instruct | 0.8050 | 0.8100 |
| Zero-shot Prompting | meta-llama/llama-4-scout-17b-16e-instruct | 0.7996 | 0.8050 |

**Finding:** LoRA fine-tuning outperforms prompting by ~7% F1 while being significantly faster and cheaper at inference time — no API calls required after training.

## Dataset

[Snappfood Sentiment Dataset](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis) — 70,000 Persian customer reviews, binary sentiment (HAPPY/SAD), balanced classes.

## Tech Stack

- **Fine-tuning:** `transformers`, `peft`, `torch`
- **Preprocessing:** `hazm`
- **Prompting baseline:** Groq API, LLaMA 4 Scout
- **Evaluation:** `scikit-learn`
- **Deployment:** `FastAPI`, `Docker`
- **Training:** Google Colab (T4 GPU)

## Project Structure

```
persian-sentiment-peft/
├── data/
│   ├── raw/                  # Original dataset
│   └── splits/               # Train/val/test CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── lora_model.py
│   │   └── prompt_baseline.py
│   ├── training/
│   │   ├── config.py
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── api/
│       ├── main.py
│       └── inference.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

### Run locally

```bash
git clone https://github.com/NaseriM1608/persian-sentiment-peft.git
cd persian-sentiment-peft
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Run with Docker

```bash
docker build -t persian-sentiment-api .
docker run -p 8000:8000 persian-sentiment-api
```

### API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "غذا عالی بود"}'
```

Response:
```json
{"label": "HAPPY", "label_id": 0}
```

## Training

Fine-tuning was done on Google Colab (T4 GPU) using LoRA with the following config:
- Rank: 8
- Alpha: 16
- Target modules: query, value
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-4
- Training time: ~23 minutes

The trained adapter is available on HuggingFace Hub: `NaseriM/persian-sentiment-lora`
