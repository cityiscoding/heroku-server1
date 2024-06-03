import pickle
import json
import random
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware

nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load trained model and other data
model = load_model("chatbot_model.keras")
with open("intents.json", "r", encoding="utf-8-sig") as file:
    intents = json.load(file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = FastAPI()

# Cấu hình CORS
origins = [
    "https://localhost:80/",  # Thêm nguồn gốc của bạn
    "http://localhost/",
    "http://localhost",  
    "http://localhost:8080",
    "http://localhost:80",
    "http://localhost/*",
    "https://pallmall.shop/*",
    "http://pallmall.shop/",
    "https://pallmall.shop",
    "http://pallmall.shop",
    "https://pallmall.shop/*",
    "http://pallmall.shop/*"     # Bao gồm cả HTTP và HTTPS nếu cần
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các headers
)

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

class Message(BaseModel):
    msg: str

@app.post("/chat/")
async def chat(input_data: Message):
    msg = input_data.msg
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return {"response": res}