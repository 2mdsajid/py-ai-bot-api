import pickle, json, random, os
from classes import Time,Date,Person,Bot,BotName,User,UserName,TrainData,Quit,Clear
from functions import clear,slow,start,greet,path,user,bot_name,bot


class StartBot:
    def start():
        file = open(path+'DataSet.json','r')
        data = json.load(file)
        
        tag_list = []
        for i in range(len(data['intents'])):
            tag_list += [data['intents'][i]['tag']]
        
        voca = open(path+'vocabulary.pickle','rb')
        vectorizer = pickle.load(voca)
        
        load = open(path+'model.pickle','rb')
        model = pickle.load(load)
        
        while True:
            inp=input(f'\n\n{user[-1].upper()} : ')
            pred = model.predict(vectorizer.transform([inp]))
            index = tag_list.index(pred[0])
            text = random.choice(data['intents'][index]['response'])
            bot_name()
            if data['intents'][index]['action']:
                cls = data['intents'][index]['cls']
                act = globals()[cls]
                act.get(text,inp)
                if cls == "TrainData":
                    StartBot.start()
            else: slow(text)

start()
StartBot.start()
