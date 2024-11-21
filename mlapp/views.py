import pickle
from django.shortcuts import render

# Load the saved model and vectorizer
with open('mlapp/sentiment_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

def predict_sentiment(request):
    if request.method == 'POST':
        user_input = request.POST['review']
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return render(request, 'result.html', {'sentiment': sentiment})
    return render(request, 'form.html')
