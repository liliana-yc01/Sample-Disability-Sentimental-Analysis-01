from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sequence = "Paralympian Ali Truwit lost her leg but still inspires us all."
candidate_labels = ["inspiration porn", "neutral", "empowering"]
print(classifier(sequence, candidate_labels))