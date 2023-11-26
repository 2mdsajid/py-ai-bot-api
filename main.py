from flask import Flask, request, jsonify
import pickle, json, random
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/get')
def get_user():
    file = open('DataSet.json','r')
    data = json.load(file)
    tag_list = []
    for i in range(len(data['intents'])):
        tag_list += [data['intents'][i]['tag']]
        
    voca = open('vocabulary.pickle','rb')
    vectorizer = pickle.load(voca)
        
    load = open('model.pickle','rb')
    model = pickle.load(load)

    msg = request.args.get('m')
    print('mmmmmmmmmmmmmmmmmmm',msg)

    pred = model.predict(vectorizer.transform([msg]))
    index = tag_list.index(pred[0])
    text = random.choice(data['intents'][index]['response'])

    return jsonify(text)

if __name__ == '__main__':
    app.run()