from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from keras.optimizers import SGD
import warnings
from googletrans import Translator, constants
warnings.filterwarnings('ignore')

translator = Translator()


sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

app = Flask(__name__)
CORS(app)

data_file = open('intents1.json').read()
intents = json.loads(data_file)

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

model = load_model("chat_model.h5")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")

    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25  
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        intent = {"intent": classes[r[0]], "probability": str(r[1])}

        for intent_data in intents['intents']:
            if intent_data['tag'] == intent['intent']:
                responses = intent_data.get('responses', [])
                links = intent_data.get('links', [])
                intent['response'] = random.choice(responses) if responses else ""
                intent['links'] = links
                break

        return_list.append(intent)

    return return_list

def answer(input_sentence):
    predictions = predict_class(input_sentence, model)
    high_prob_prediction = next((p for p in predictions if float(p["probability"]) > 0.7), None)

    if high_prob_prediction:
        print("Intent:", high_prob_prediction["intent"])
        print("Probability:", high_prob_prediction["probability"])
        print("Response:", high_prob_prediction["response"])
        print("Links:", high_prob_prediction["links"])
        return high_prob_prediction["response"]
    else:
        print("No high probability intent found.")
        return "I am sorry, I couldn't Understand that."
    


@app.route('/get_response', methods=['POST'])
def api():
    if request.method == 'POST':
        data = request.json
        question = translator.translate(data['question'])
        data_ans = answer(question.text)
        dest = "en"
        print(data)
        if data['language']=="Hindi":
            dest = "hi"
        elif data['language']=="Marathi":
            dest = "mr"
        data_ans = translator.translate(data_ans,dest=dest).text
        response_data = {'message': 'Received your data!', 'answer': data_ans}
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
