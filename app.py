from flask import Flask, request, jsonify
import joblib
import re
import string

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_sms_detect_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Save and load vectorizer too

# Text cleaning function (same as used during training)
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess and vectorize
        cleaned_text = clean_text(text)
        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]
        result = {"spam": bool(prediction)}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
