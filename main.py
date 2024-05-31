from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import json
import numpy as np
import random
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

# Cấu hình CORS
origins = [
    "https://pallmall.shop",  # Thêm nguồn gốc của bạn
    "http://pallmall.shop"    # Bao gồm cả HTTP và HTTPS nếu cần
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.keras')
intents = json.loads(open('intents.json', encoding='utf-8-sig').read())
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

@app.post("/chat/")
async def chat(input_data: dict):
    msg = input_data.get("msg")
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return {"response": res}
