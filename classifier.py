from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="kertys/yelp_review_classifier")
print("Введите текст отзыва(английский язык даёт лучшие результаты)")
review = input()
print(classifier(review))