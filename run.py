import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  # Khởi tạo đối tượng lemmatizer
lemmatizer = WordNetLemmatizer()

# Các hàm và biến global khác


app = Flask(__name__)

# Load model and other necessary data
model = load_model('chatbot_model.keras')
with open('intents.json', encoding='utf-8-sig') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Tìm thấy trong bag: {w}") 
    return np.array(bag)

def predict_class(sentence, model):
    bow_sentence = bow(sentence, words, show_details=False)
    res = model.predict(np.array([bow_sentence]))[0]
    ERROR_THRESHOLD = 0.15
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "Tôi không hiểu ý bạn. Bạn có thể hỏi lại không?"
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

@app.route('/', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        message = request.form['message']
        response = chatbot_response(message)
        return jsonify({"response": response})
    else:
        return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    message = request.json['message']
    response = chatbot_response(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
