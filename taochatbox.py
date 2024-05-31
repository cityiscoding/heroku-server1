import nltk
nltk.download('punkt')  # Tải xuống dữ liệu 'punkt' (bộ tách từ và câu) từ NLTK
nltk.download('wordnet')  # Tải xuống dữ liệu 'wordnet' (cơ sở dữ liệu từ vựng) từ NLTK
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  # Khởi tạo đối tượng lemmatizer
import json
import pickle
from tkinter import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []  # Danh sách tất cả các từ
classes = []  # Danh sách các lớp (ý định)
documents = []  # Danh sách các cặp (câu mẫu, ý định)
ignore_words = ['?', '!']  # Danh sách các từ cần loại bỏ

# Đọc file dữ liệu huấn luyện (intents.json)
data_file = open('intents.json', encoding='utf-8-sig').read()
intents = json.loads(data_file)

# Xử lý dữ liệu huấn luyện
for intent in intents['intents']:  # Duyệt qua từng ý định
    for pattern in intent['patterns']:  # Duyệt qua từng câu mẫu
        # Tách từ (tokenize) từ câu mẫu
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # Thêm các từ vào danh sách words
        # Thêm cặp (câu mẫu, ý định) vào danh sách documents
        documents.append((w, intent['tag']))

        # Thêm ý định vào danh sách classes nếu chưa có
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Chuẩn hóa dữ liệu: đưa về dạng gốc (lemmatize), chuyển thành chữ thường và loại bỏ các từ trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sắp xếp danh sách classes
classes = sorted(list(set(classes)))

# In ra số lượng documents, classes, words
# print(len(documents), "documents")
# print(len(classes), "classes", classes)
# print(len(words), "unique lemmatized words", words)

# Lưu danh sách words và classes vào file pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Tạo dữ liệu huấn luyện
training = []
# Tạo mảng rỗng cho đầu ra (output)
output_empty = [0] * len(classes)

# Tạo tập huấn luyện (training set), bag of words cho mỗi câu
for doc in documents:
    # Khởi tạo bag of words
    bag = []
    # Danh sách các từ đã được tách từ câu mẫu
    pattern_words = doc[0]
    # Đưa các từ về dạng gốc (lemmatize) và chuyển thành chữ thường
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Tạo mảng bag of words với giá trị 1 nếu từ có trong pattern_words, ngược lại là 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Đầu ra là mảng '0' cho mỗi nhãn và '1' cho nhãn hiện tại
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Kết hợp bag of words và output_row thành một danh sách và thêm vào dữ liệu huấn luyện
    training.append(bag + output_row)

# Xáo trộn dữ liệu huấn luyện và chuyển thành mảng numpy
random.shuffle(training)
training = np.array(training)

# Tách dữ liệu thành train_x (mẫu) và train_y (nhãn)
train_x = list(training[:, :len(words)])
train_y = list(training[:, len(words):])
print("Training data created")  # Dữ liệu huấn luyện đã được tạo

# Tạo mô hình 3 lớp: 128 neurons (lớp 1), 64 neurons (lớp 2), số neurons bằng số ý định (lớp 3)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Biên dịch mô hình. Sử dụng SGD với Nesterov accelerated gradient
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện và lưu mô hình
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)

print("Model created")  # Mô hình đã được tạo
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# Thử nghiệm trên giao diện tkinter
# from keras.models import load_model
# import json
# import random
# import pickle
# import nltk
# import numpy as np

# # Tải xuống các gói dữ liệu cần thiết cho NLTK
# nltk.download('punkt')   
# nltk.download('wordnet')

# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# # Tải mô hình chatbot đã được huấn luyện trước đó
# model = load_model('chatbot_model.keras')

# # Đọc dữ liệu từ file intents.json
# with open('intents.json', encoding='utf-8-sig') as file:
#     intents = json.load(file)

# # Tải các từ vựng và lớp (ý định) đã được lưu từ quá trình huấn luyện
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# def clean_up_sentence(sentence):
#     """Tách câu thành các từ và đưa chúng về dạng gốc (lemmatize)."""
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=True):
#     """Chuyển câu thành dạng bag of words (mảng các số 0 và 1)."""
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)  # Khởi tạo mảng bag toàn 0
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1  # Nếu từ có trong từ điển, đánh dấu 1 tại vị trí tương ứng
#                 if show_details:
#                     print(f"Tìm thấy trong bag: {w}") 
#     return np.array(bag)  # Trả về mảng NumPy

# def predict_class(sentence, model):
#     """Dự đoán ý định của câu và trả về danh sách các ý định có xác suất cao."""
#     bow_sentence = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([bow_sentence]))[0]
#     ERROR_THRESHOLD = 0.15  # Ngưỡng xác suất tối thiểu để chấp nhận ý định
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo xác suất giảm dần
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

# def getResponse(ints, intents_json):
#     """Trả về câu trả lời ngẫu nhiên từ các câu trả lời liên quan đến ý định được dự đoán."""
#     if ints:  # Kiểm tra nếu có ít nhất một ý định được dự đoán
#         tag = ints[0]['intent']
#         list_of_intents = intents_json['intents']
#         for i in list_of_intents:
#             if i['tag'] == tag:
#                 result = random.choice(i['responses'])
#                 break
#     else:  # Nếu không có ý định nào được dự đoán, trả về câu mặc định
#         result = "Tôi không hiểu ý bạn. Bạn có thể hỏi lại không?"
#     return result

# # ... (Phần giao diện tkinter giữ nguyên)
# def chatbot_response(text):
#     ints = predict_class(text, model)
#     res = getResponse(ints, intents)
#     return res
# from tkinter import *

# BG_GRAY = "#ABB2B9"
# BG_COLOR = "#c5f0e3"
# TEXT_COLOR = "#000000"


# FONT = "Helvetica 14"
# FONT_BOLD = "Helvetica 13 bold"


# def send(event):
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     EntryBox.delete("0.0",END)
#     if msg != '':
#         ChatLog.config(state=NORMAL)
#         ChatLog.insert(END, "Bạn: " + msg + '\n\n')
#         ChatLog.config(foreground="#000000", font=("Verdana", 12 ))

#         res = chatbot_response(msg)
#         ChatLog.insert(END, "Nhân viên: " + res + '\n\n')

#         ChatLog.config(state=DISABLED)
#         ChatLog.yview(END)
        
        
# base = Tk()
# base.title("E-Commerce Chatbot")
# base.resizable(width=FALSE, height=FALSE)
# base.configure(width=800, height=800, bg=BG_COLOR)


# #Create Chat window
# ChatLog = Text(base, bd=0, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_BOLD)
# ChatLog.config(state=DISABLED)

# head_label = Label(base, bg=BG_COLOR, fg=TEXT_COLOR, text="Chatbox trả lời tự động", font=FONT_BOLD, pady=10)
# head_label.place(relwidth=1)

# line = Label(base, width=450, bg=BG_GRAY)


# #Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set
# ChatLog.focus()

# #Create Button to send message
# SendButton = Button(base, font=("Verdana", 12,'bold'), text="Gửi", width="12", height=15,
#                     bd=0, bg="#ed9061", activebackground="#3c9d9b",fg='#ffffff',
#                     command=lambda: send)

# #Create the box to enter message
# EntryBox = Text(base, bg="white",width="29", height="5", font="Arial", background="#dddddd")
# EntryBox.focus()
# EntryBox.bind("<Return>", send)

# scrollbar.place(x=775,y=6, height=800)
# line.place(x=0,y=35, height=1, width=770)
# ChatLog.place(x=5,y=40, height=700, width=770)
# EntryBox.place(x=0, y=740, height=60, width=600)
# SendButton.place(x=600, y=740, height=60, width=175)
# base.mainloop()