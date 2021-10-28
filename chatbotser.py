
from tkinter import *
import time
import tkinter.messagebox
import threading
import speech_recognition as sr

import json 
from tensorflow import keras
import librosa
import soundfile
import tensorflow as tf
import random
import pickle
import sounddevice as sd

from scipy.io.wavfile import write


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model1 = load_model('chatbot_model.h5')
import json
import random
with open('C:/Users/Akhil/Desktop/chatbot/intents1.json',encoding="utf8") as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model1)
    res = getResponse(ints, intents)
    return res


with open('C:/Users/Akhil/Desktop/ser/serintents.json',encoding="utf8") as file:
    data = json.load(file)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
    if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


model = pickle.load(open("sermodel_pickle", 'rb'))
# x_int=set_audio()
# x=model.predict(x_int.reshape(1,-1))
# print(x)
saved_username = ["You"]

window_size="400x400"

class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        # sets default bg for top level windows
        self.tl_bg =  "#1c2e44"
        self.tl_bg2 ="#263b54"
        self.tl_fg ="#FFFFFF"
        self.font = "Arial 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)
# Menu bar

    # File
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
       # file.add_command(label="Save Chat Log", command=self.save_chat)
        file.add_command(label="Clear Chat", command=self.clear_chat)
      #  file.add_separator()
        file.add_command(label="Exit",command=self.chatexit)
    
     # username
  
        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        #help_option.add_command(label="Features", command=self.features_msg)
        help_option.add_command(label="About PyBot", command=self.msg)
        help_option.add_command(label="Develpoers", command=self.about)

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        # scrollbar for text box
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        # contains messages
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # frame containing user entry field
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # entry field
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)
        # self.users_message = self.entry_field.get()

        # frame containing send button and emoji button
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        # send button
        self.send_button = Button(self.send_button_frame, text="Send", width=5, relief=GROOVE, bg='white',bd=1,
                                   command=lambda: self.send_message_insert(None), activebackground="#1c2e44",
                                  activeforeground="#FFFFFF")
        self.send_button.pack(side=LEFT, ipady=8)
        self.master.bind("<Return>", self.send_message_insert)
        
        self.last_sent_label(date="No messages sent.")
        #speech button
        
        self.speech_button = Button(self.send_button_frame, text="Speech", width=5, relief=GROOVE, bg='white',
                                  bd=1, command=lambda:self.speech(), activebackground="#263b54",
                                  activeforeground="#000000")
        self.speech_button.pack(side=LEFT, ipady=8)
        
        #refresh
       
        
        self.master.config(bg="#263b54")
        self.text_frame.config(bg="#263b54")
        self.text_box.config(bg="light yellow", fg="black")
        self.entry_frame.config(bg="#263b54")
        self.entry_field.config(bg="#263b54", fg="white", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#263b54")
        self.send_button.config(bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        self.speech_button.config(bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#263b54", fg="#FFFFFF")

        self.tl_bg = "#1c2e44"
        self.tl_bg2 = "#263b54"
        self.tl_fg = "#FFFFFF"

        #t2 = threading.Thread(target=self.send_message_insert(, name='t1')
        #t2.start()
        
    
    # def playResponce(self,responce,langg):
    #     speak = gTTS(text=responce, lang=langg, slow= False)  
    #     speak.save("captured_voice.mp3")     
    #     playsound('.\captured_voice.mp3')
    #     os.remove("captured_voice.mp3")
        
        
    def last_sent_label(self, date):

        try:
            self.sent_label.destroy()
        except AttributeError:
            pass

        self.sent_label = Label(self.entry_frame, font="Arial 7", text=date, bg=self.tl_bg2, fg=self.tl_fg)
        self.sent_label.pack(side=LEFT, fill=X, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()

   
    def msg(self):
        tkinter.messagebox.showinfo('Counselling chatbot')

    def about(self):
        tkinter.messagebox.showinfo("PyBOT Developers","1.Akhil")
    def send_message_bot(self,message):
        ans=self.chat(message)
        
        if(ans==None):
            ans="See you again..."
        print(ans)
        pr="Bot : " + ans + "\n"
        self.text_box.configure(state=NORMAL)
       
        self.text_box.insert(END, pr)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.last_sent_label(str(time.strftime( "Lastmessage sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        self.entry_field.delete(0,END)
    


    
    def send_message_insert(self, message):
        user_input = self.entry_field.get()
        if(user_input==''):
            pr1 = "You: " + message + "\n"
            query=message
            self.text_box.configure(state=NORMAL)
            self.text_box.insert(END, pr1)
            res = chatbot_response(user_input)
            self.text_box.insert(END, "Bot: " + res + '\n\n')
            self.text_box.configure(state=DISABLED)
            self.text_box.see(END)
        else:
            pr1 = "You: " + user_input + "\n"
            query=user_input
            self.text_box.configure(state=NORMAL)
            self.text_box.insert(END, pr1)
            res = chatbot_response(user_input)
            self.text_box.insert(END, "Bot: " + res + '\n\n')
            
            self.text_box.configure(state=DISABLED)
            self.text_box.see(END)
            msg = self.text_box.get("1.0",'end-1c').strip()
            self.text_box.delete("0.0",END)

        
        
  
            
    
    def chat(self,inp):
        list_of_intents = data['intents']
        x=model.predict(inp.reshape(1,-1))
        for i in list_of_intents:
            if(i['tag']== x[0]):
                return random.choice(i['responses'])
                
        else:
            return "no defined emotion recognized" 
        
    def set_audio(self):
        t=sr.Recognizer()
        feature=extract_feature("C:/Users/Akhil/Desktop/ser/New folder/Actor_03/03-01-03-02-01-02-03.wav", mfcc=True, chroma=True, mel=True)   
        with sr.AudioFile("C:/Users/Akhil/Desktop/ser/New folder/Actor_03/03-01-03-02-01-02-03.wav") as source:   
            audio = t.listen(source)    
        try:
            mes=t.recognize_google(audio) 
            self.send_message_insert(mes)
            self.send_message_bot(feature) 
        except LookupError:                                
            print("Could not understand audio")
        
          
 
    def speech(self):
        self.ins_label = Label(self.text_box, font="Verdana 7", text="taking audio input to recognizing emotion ", bg=self.tl_bg2, fg=self.tl_fg)
        self.ins_label.pack(side=BOTTOM, fill=X)
        def callback():
            fs=16000
            seconds=3
            myrecording =sd.rec(int(seconds *fs),samplerate=fs,channels=1,dtype=np.int16)
            sd.wait()
            write("C:/Users/Akhil/Desktop/ser/output.wav",fs,myrecording)
            t=sr.Recognizer()
            self.ins_label.destroy()
            try:
                feature=extract_feature("C:/Users/Akhil/Desktop/ser/output.wav", mfcc=True, chroma=True, mel=True)   
                with sr.AudioFile("C:/Users/Akhil/Desktop/ser/output.wav") as source:   
                    audio = t.listen(source) 
                mes=t.recognize_google(audio)
                self.send_message_insert(mes)
                self.send_message_bot(feature) 
         
            
        
            # t=sr.Recognizer()
            # try:
            #     feature=extract_feature("C:/Users/Akhil/Desktop/ser/New folder/Actor_01/03-01-01-01-01-01-01.wav", mfcc=True, chroma=True, mel=True)   
            #     with sr.AudioFile("C:/Users/Akhil/Desktop/ser/New folder/Actor_01/03-01-01-01-01-01-01.wav") as source:   
            #         audio = t.listen(source)    
            #     mes=t.recognize_google(audio)
            #     self.send_message_insert(mes)
            #     self.send_message_bot(feature)
            
        
 

              
            except sr.RequestError as e: 
                print("Could not request results; {0}".format(e))            
          
            except sr.UnknownValueError: 
                print("no query")
                
        
        t = threading.Thread(target=callback)
        t.start()
        
root=Tk()
a = ChatInterface(root)
root.geometry(window_size)
root.title("Bot")
root.mainloop()


