import transformers
from transformers import pipeline

# Download the model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Save the model to a file
classifier.save_pretrained("models/hugging_face/distilbert-base-uncased-finetuned-sst-2-english")