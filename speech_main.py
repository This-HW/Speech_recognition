# path
import os
from os.path import isdir, join
from pathlib import Path


# Scientific Math
import numpy as np
import scipy
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf


# Deep learning
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import random
import copy

# MFCC & audio load
import librosa
import subprocess

# DataSize 880 =
size = 880


def start():
    '''
    Main Function, start

    Parameters
    ----------
        None

    Returns
    -------
        None

    Note
    ----
        Save Model(h5, json)

    '''


    # DATA PATH SET
    train_path = '.\\trans_data\\train\\'
    print(os.listdir(train_path))
    dirs = [f for f in os.listdir(train_path) if isdir(join(train_path, f))]
    dirs.sort()
    print('Number of labels: ' + str(len(dirs)))
    print(dirs)

    all_wav = []
    unknown_wav = []
    label_all = []
    label_value = {}
    target_list = ['up', 'down',  'on', 'off', 'house', 'stop']
    unknown_list = [d for d in dirs if d not in target_list ]
    print('target_list : ', end='')
    print(target_list)
    print('unknowns_list : ', end='')
    print(unknown_list)


    i = 0
    a = 0

    hop_length = 512 # 23ms
    for direct in dirs[0:]:
        waves = [f for f in os.listdir(join(train_path, direct)) if f.endswith('.wav')]
        label_value[direct] = i
        i = i + 1
        print(str(i) + ":" + str(direct) + " ", end="")
        for wav in waves:
            samples_, sample_rate = librosa.load(join(join(train_path, direct), wav))
            # MFCC
            samples = librosa.feature.mfcc(y=samples_, sr=sample_rate, hop_length=hop_length)
            samples = np.array(samples)
            samples = samples.flatten()

            if len(samples) != size:
                a += 1
                continue
            if direct in unknown_list:
                unknown_wav.append(samples)
            else:
                label_all.append(direct)
                all_wav.append([samples, direct])


    #Resize data to fit our model
    print("float(a)/len(waves):::",float(a)/len(waves))
    wav_all = np.reshape(np.delete(all_wav, 1, 1), (len(all_wav)))
    label_all = [i for i in np.delete(all_wav, 0, 1).tolist()]
    print(wav_all.shape)

    wav_vals = np.array([x for x in wav_all])
    label_vals = [x for x in label_all]
    print(wav_vals.shape)

    wav_vals    = np.reshape(wav_vals, (-1, size))
    print(len(wav_vals))
    print(len(label_vals))
    train_wav, test_wav, train_label, test_label = train_test_split(wav_vals, label_vals,
                                                                        test_size = 0.2,
                                                                        random_state = 2019,
                                                                        shuffle = True)

    # Parameters
    lr = 0.001
    batch_size = 256
    drop_out_rate = 0.5
    input_shape = (size, 1)

    #For Conv1D add Channel
    train_wav = train_wav.reshape(-1, size, 1)
    test_wav = test_wav.reshape(-1, size, 1)

    label_value = target_list
    label_value.append('silence')

    new_label_value = dict()
    for i, l in enumerate(label_value):
        new_label_value[l] = i
    label_value = new_label_value

    # Make Label data 'string' -> 'class num'
    temp = []
    for v in train_label:
        temp.append(label_value[v[0]])
    train_label = np.array(temp)

    temp = []
    for v in test_label:
        temp.append(label_value[v[0]])
    test_label = np.array(temp)

    # Make Label data 'class num' -> 'One hot vector'
    train_label = keras.utils.to_categorical(train_label, len(label_value))
    test_label = keras.utils.to_categorical(test_label, len(label_value))
    print('Train_Wav Demension : ' + str(np.shape(train_wav)))
    print('Train_Label Demension : ' + str(np.shape(train_label)))
    print('Test_Wav Demension : ' + str(np.shape(test_wav)))
    print('Test_Label Demension : ' + str(np.shape(test_label)))
    print('Number Of Labels : ' + str(len(label_value)))
    # Conv1D Model
    input_tensor = Input(shape=(input_shape))


    # 1D
    x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    output_tensor = layers.Dense(len(label_value), activation='softmax')(x)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr = lr),
                 metrics=['accuracy'])
    model.summary()
    history = model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
              batch_size=batch_size,
              epochs=500,
              verbose=1)


    # show plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Model save
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def modelLoad():
    # model load
    if os.path.isfile("./model.json"):
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr = 0.001),
                             metrics=['accuracy'])
        return loaded_model

    else:
        print("model.json not exist. learning start.")
        start()

def pred(model,file_path):
    samples, sample_rate = librosa.load(file_path)
    samples = librosa.feature.mfcc(y=samples, sr=sample_rate)
    samples = np.array(samples)
    samples = samples.flatten()[:size]

    # (-1, size, 1)
    samples = samples.reshape(-1, 880, 1)
    pred = model.predict(samples)
    pred_cls = np.argmax(pred, axis=1)[0] + 1
    if pred_cls==1:
        print("up")
    if pred_cls == 2:
        print("down")
    if pred_cls==3:
        print("on")
    if pred_cls==4:
        print("off")
    if pred_cls==5:
        print("house")
    if pred_cls==6:
        print("stop")
    if pred_cls==7:
        print("명령어가 아닙니다.")
        showwarning("warning","명령어가 아닙니다")

    return pred_cls


if __name__=="__main__":
    from tkinter import filedialog
    from tkinter import *
    from tkinter.messagebox import showwarning
    from PIL import ImageTk, Image
    chk_on = -1
    updown=True
    onoff=False
    
    def up():
        global img1, chk_on, updown, onoff
        if chk_on==-1:
            return
        
        updown=True
        if chk_on==1 and onoff==False:
            tkim=Image.open("slide3.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif chk_on==1 and onoff==True:
            tkim=Image.open("slide4.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        else: return
    
    def down():
        global img1, chk_on, updown, onoff
        if chk_on==-1:
            return
        
        updown=False
        if chk_on==1 and onoff==False:
            tkim=Image.open("slide1.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif chk_on==1 and onoff==True:
            tkim=Image.open("slide2.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        else: return
    
    def on():
        global img1, chk_on, updown, onoff
        if chk_on==-1:
            return
        
        onoff=True
        if chk_on==1 and updown==False:
            tkim=Image.open("slide2.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif chk_on==1 and updown==True:
            tkim=Image.open("slide4.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        else: return
    
    def off():
        global img1, chk_on, updown, onoff
        if chk_on==-1:
            return
        
        onoff=False
        if chk_on==1 and updown==False:
            tkim=Image.open("slide1.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif chk_on==1 and updown==True:
            tkim=Image.open("slide3.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        else: return
    
    def powerOff():
        global img1, chk_on
        chk_on=-1
        tkim =Image.open("power_off.jpg")
        tkim = tkim.resize((700, 500), Image.ANTIALIAS)
        tkim = ImageTk.PhotoImage(tkim)
        img1.config(image=tkim)
        img1.image = tkim
        
    def powerOn():
        global img1, chk_on, updown, onoff
        chk_on=1
        if onoff==False and updown==False:
            tkim=Image.open("slide1.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif onoff==True and updown==True:
            tkim=Image.open("slide4.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif onoff==False and updown==True:
            tkim=Image.open("slide3.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        elif onoff==True and updown==False:
            tkim=Image.open("slide2.jpg")
            tkim=tkim.resize((700,500), Image.ANTIALIAS)
            tkim=ImageTk.PhotoImage(tkim)
            img1.config(image=tkim)
            img1.image=tkim
            
        else: return


    def wavopen():

        pred_cls = 0
        try:
            file = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        except:
            print("파일 로드 실패")


        pred_cls = pred(model, file)

        if pred_cls==1:
            up()
        if pred_cls==2:
            down()
        if pred_cls==3:
            on()
        if pred_cls==4:
            off()
        if pred_cls==5:
            powerOn()
        if pred_cls==6:
            powerOff()


    model = modelLoad()
    window = Tk()


    
    img = Image.open("power_off.jpg")
    img = img.resize((700, 500), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img1 = Label(window, image=img)
    img1.place(x=150, y=0)    
    
    window.title("SPEECH COMMAND")
    window.geometry("840x500+100+100")
    window.resizable(False, False)

    button1 = Button(window, text="wav file load", overrelief="solid", width=15,height=2,
                    command=wavopen, repeatdelay=1000, repeatinterval=100)

    button1.grid(row=1, column=1)
    button1.place(x=20,y=20)

    window.mainloop()




