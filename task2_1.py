# library import
import sys
import json
import os
import time
import wave
import re
from tkinter import *
from tkinter import filedialog as fd
from vosk import Model, KaldiRecognizer, SpkModel
import wave
import numpy as np
import IPython
import librosa as li
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import stats
import argparse
from pydub import AudioSegment

# Tkinter description
window=Tk()
fileExist = 0
textOfSound = Text(window, height = 10)
textOfSound.pack(side=TOP, fill = BOTH, expand = TRUE)  

# function of transcribation to display text
def doTranscipt(): 
    global fileExist
    global fileName
    if (fileExist):           
        workdir = 'C:/Users/chess/Downloads'
        model = Model(workdir + "/vosk-model-ru-0.22")        
        wavfile = filename 
        #inputfile = workdir + '/CALL_20211230_140757.m4a' #44100 mono 96kbs
        #inputfile = workdir + '/01.Alehin-Zasedanie.mp3'        
        #-y (global) Overwrite output files without asking.
        #!ffmpeg -y -i $inputfile -ar 16000 -ac 2 -ab 192K -f wav $wavfile
        #!ffmpeg -y -i $inputfile -ar 32000 -ac 2 -ab 192K -f wav $wavfile
        #!ffmpeg -y -i $inputfile -ar 48000 -ac 2 -ab 192K -f wav $wavfile
        #!ffmpeg -y -i $inputfile -ar 48000 -ac 1 -ab 192K -f wav $wavfile
        !ffmpeg -y -i $inputfile -ar 48000 -ac 1 -f wav $wavfile
        #small: 4 мин. 16kbps wav file - за 15 сек, 32kbps - 25s, 48kbps - 58 сек, 48-96kbps - 17sec!, 48-144kbps - 16sec!
        #big: 48-96kbps stereo - 65-75sec
        #big: 48-48kbps mono - 64-..sec
        wf = wave.open(wavfile, "rb")
        rcgn_fr = wf.getframerate() * wf.getnchannels()
        rec = KaldiRecognizer(model, rcgn_fr)
        result = ''
        last_n = False
        #read_block_size = 4000 
        read_block_size = wf.getnframes()
        while True: #Можно читать файл блоками, тогда можно выводить распознанный текст частями, но слова на границе блоков могут быть распознаны некорректно
            data = wf.readframes(read_block_size)
            if len(data) == 0:
                break       
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())            
                if res['text'] != '':
                    result += f" {res['text']}"
                    if read_block_size < 200000:
                        print(res['text'] + " \n")                    
                    last_n = False
                elif not last_n:
                    result += '\n'
                    last_n = True      
        res = json.loads(rec.FinalResult())
        result += f" {res['text']}"
        textOfSound.delete('1.0', END)
        textOfSound.insert(END, '\n'.join(line.strip() for line in re.findall(r'.{1,150}(?:\s+|$)', result)))        
    else:
        textOfSound.delete('1.0', END)
        textOfSound.insert(END, "There is no enough information")    
btnTranscript = Button(window, text="Transcript", fg='blue', height = 5, command = doTranscipt)
btnTranscript.pack(side=BOTTOM, fill = BOTH)

# function to open file and save file name
def saveFileName():
    global fileExist
    global filename 
    filetypes = (
        ('wav', '*.wav'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)    
    if (filename != ''):
        fileExist = 1
    else:
        fileExist = 0
 
# button of load music
btnLoad = Button(window, text="Load", fg='blue', height = 5, command = saveFileName)   
btnLoad.pack(side=TOP, fill = BOTH, expand = TRUE)

# did diarization 2 variant
def doDiarizate2():
    global filename
    global fileExist
    print(filename)
    if (fileExist):
        sound = AudioSegment.from_wav(filename)
        sound = sound.set_channels(1)
        sound.export(filename, format="wav")
        workdir = 'C:/Users/chess/Downloads'
        model_path = workdir + "/vosk-model-ru-0.22"
        spk_model_path = workdir + "/vosk-model-spk-0.4"
        print(model_path)
        print(spk_model_path)
        # Large vocabulary free form recognition
        model = Model(model_path)
        spk_model = SpkModel(spk_model_path)       
        wavfile = filename
        #wavfile = workdir + '/test.wav'
        wf = wave.open(wavfile, "rb")
        if wf.getnchannels() != 1:
            print ("Audio file must be mono.")
            exit (1)
        if wf.getsampwidth() != 2:
            print ("Audio file must be WAV format PCM. sampwidth=", wf.getsampwidth())
            exit (1)     
        if wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format PCM. comptype=", wf.getcomptype())
            exit (1)
        # We compare speakers with cosine distance. We can keep one or several fingerprints for the speaker in a database
        # to distingusih among users.
        spk_sig = [-1.110417,0.09703002,1.35658,0.7798632,-0.305457,-0.339204,0.6186931,-0.4521213,0.3982236,-0.004530723,0.7651616,0.6500852,-0.6664245,0.1361499,0.1358056,-0.2887807,-0.1280468,-0.8208137,-1.620276,-0.4628615,0.7870904,-0.105754,0.9739769,-0.3258137,-0.7322628,-0.6212429,-0.5531687,-0.7796484,0.7035915,1.056094,-0.4941756,-0.6521456,-0.2238328,-0.003737517,0.2165709,1.200186,-0.7737719,0.492015,1.16058,0.6135428,-0.7183084,0.3153541,0.3458071,-1.418189,-0.9624157,0.4168292,-1.627305,0.2742135,-0.6166027,0.1962581,-0.6406527,0.4372789,-0.4296024,0.4898657,-0.9531326,-0.2945702,0.7879696,-1.517101,-0.9344181,-0.5049928,-0.005040941,-0.4637912,0.8223695,-1.079849,0.8871287,-0.9732434,-0.5548235,1.879138,-1.452064,-0.1975368,1.55047,0.5941782,-0.52897,1.368219,0.6782904,1.202505,-0.9256122,-0.9718158,-0.9570228,-0.5563112,-1.19049,-1.167985,2.606804,-2.261825,0.01340385,0.2526799,-1.125458,-1.575991,-0.363153,0.3270262,1.485984,-1.769565,1.541829,0.7293826,0.1743717,-0.4759418,1.523451,-2.487134,-1.824067,-0.626367,0.7448186,-1.425648,0.3524166,-0.9903384,3.339342,0.4563958,-0.2876643,1.521635,0.9508078,-0.1398541,0.3867955,-0.7550205,0.6568405,0.09419366,-1.583935,1.306094,-0.3501927,0.1794427,-0.3768163,0.9683866,-0.2442541,-1.696921,-1.8056,-0.6803037,-1.842043,0.3069353,0.9070363,-0.486526]
        #spk_sig =[-0.435445, 0.877224, 1.072917, 0.127324, -0.605085, 0.930205, 0.44148, -1.20399, 0.069384, 0.538427, 1.226569, 0.852291, -0.806415, -1.157439, 0.313101, 1.332273, -1.628154, 0.402829, 0.472996, -1.479501, -0.065581, 1.127467, 0.897095, -1.544573, -0.96861, 0.888643, -2.189499, -0.155159, 1.974215, 0.277226, 0.058169, -1.234166, -1.627201, -0.429505, -1.101772, 0.789727, 0.45571, -0.547229, 0.424477, -0.919078, -0.396511, 1.35064, -0.02892, -0.442538, -1.60219, 0.615162, 0.052128, -0.432882, 1.94985, -0.704909, 0.804217, 0.472941, 0.333696, 0.47405, -0.214551, -1.895343, 1.511685, -1.284075, 0.623826, 0.034828, -0.065535, 1.604209, -0.923321, 0.502624, -0.288166, 0.536349, -0.631745, 0.970297, 0.403614, 0.131859, 0.978622, -0.5083, -0.104544, 1.629872, 1.730207, 1.010488, -0.866015, -0.711263, 2.359106, 1.151348, -0.426434, -0.80968, -1.302515, -0.444948, 0.074877, 1.352473, -1.007743, 0.318039, -1.532761, 0.145248, 3.59333, -0.467264, -0.667231, -0.890853, -0.197016, 1.546726, 0.890309, -0.7503, 0.773801, 0.84949, 0.391266, -0.79776, 0.895459, -0.816466, 0.110284, -1.030472, -0.144815, 1.087008, -1.448755, 0.776005, -0.270475, 1.223657, 1.09254, -1.237237, 0.065166, 1.487602, -1.409871, -0.539695, -0.758403, 0.31941, -0.701649, -0.210352, 0.613223, 0.575418, -0.299141, 1.247415, 0.375623, -1.001396]
        def cosine_dist(x, y):
            nx = np.array(x)
            ny = np.array(y)
            return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
        rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels(), spk_model)
        #rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels())
        rec.SetSpkModel(spk_model)     
        #res={};
        wf.rewind()
        #while True:    
        for i in range(1080):
            data = wf.readframes(4000)
            datalen=len(data);
            if datalen == 0:
                res = json.loads(rec.FinalResult())
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                print ("Text:", res['text'])
                if 'spk' in res:
                    print ("X-vector:", res['spk'])
                    print ("Speaker distance:", cosine_dist(spk_sig, res['spk']), end=' ')
                    print ("based on frames:", res['spk_frames'])
            if datalen == 0:
                break
        #Note that second distance is not very reliable because utterance is too short. Utterances longer than 4 seconds give better xvector
    else:
        print('nonono')       
 
# did diarization 1 variant
def doDiarizate():
    global filename
    global fileExist
    if (fileExist):   
        file_name = filename
    
        audio_time_series, sample_rate = li.load(file_name)
        length_series = len(audio_time_series)
        print("length_series: ", length_series)
        print("sample_rate: ", sample_rate)
        print("audio_time_series: ", audio_time_series)
        print(audio_time_series)
        
        zero_crossings = []
        energy = []
        entropy_of_energy = []
        mfcc = []
        chroma_stft = []
        for i in range(0,length_series,int(sample_rate/5.0)):
             frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
             z = li.zero_crossings(frame_self)
             arr = np.nonzero(z)
             zero_crossings.append(len(arr[0]))
             #e = li.feature.rmse(frame_self)
             e = li.feature.rms(frame_self)
             energy.append(np.mean(e))
             ent = 0.0
             m = np.mean(e)
             for j in range(0,len(e[0])):
                  q = np.absolute(e[0][j] - m)
                  ent = ent + (q * np.log10(q))
             entropy_of_energy.append(ent)
             mt = []
             mf = li.feature.mfcc(frame_self)
             for k in range(0,len(mf)):
                  mt.append(np.mean(mf[k]))
             mfcc.append(mt)
             ct = []
             cf = li.feature.chroma_stft(frame_self)
             for k in range(0,len(cf)):
                  ct.append(np.mean(cf[k]))
             chroma_stft.append(ct)
             #print(i)
        f_list_1 = []
        f_list_1.append(zero_crossings)
        f_list_1.append(energy)
        f_list_1.append(entropy_of_energy)
        f_np_1 = np.array(f_list_1)
        f_np_1 = np.transpose(f_np_1)
        sp_centroid = []
        sp_bandwidth = []
        sp_contrast = []
        sp_rolloff = []
        for i in range(0,length_series,int(sample_rate/5.0)):
             frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
             cp = li.feature.spectral_centroid(y=frame_self, hop_length=220500)
             sp_centroid.append(cp[0][0])
             bp = li.feature.spectral_bandwidth(y=frame_self, hop_length=220500)
             sp_bandwidth.append(bp[0][0])
             csp = li.feature.spectral_contrast(y=frame_self, hop_length=220500)
             sp_contrast.append(np.mean(csp))
             rsp = li.feature.spectral_rolloff(y=frame_self, hop_length=220500)
             sp_rolloff.append(np.mean(rsp[0][0]))
             #print(i)    
        f_list_2 = []
        f_list_2.append(sp_centroid)
        f_list_2.append(sp_bandwidth)
        f_list_2.append(sp_contrast)
        f_list_2.append(sp_rolloff)
        f_np_2 = np.array(f_list_2)
        f_np_2 = np.transpose(f_np_2)     
        f_np_3 = np.array(mfcc)
        f_np_4 = np.array(chroma_stft)     
        master = np.concatenate([f_np_1, f_np_2, f_np_3, f_np_4], axis=1)      
        #cluster_obj = AffinityPropagation().fit(master)
        cluster_obj = KMeans(n_clusters = 2 ,random_state=0).fit(master)
        print("Number of clusters : " + str(len(cluster_obj.cluster_centers_)))#cluster_centers_indices_
        res = cluster_obj.predict(master)
        print(cluster_obj.get_params())
        s = res[0]
        t=0.0
        time = []
        speaker = []
        time.append(t)
        speaker.append(s)
        for u in range(0, len(res), 1):
             if(res[u]==s):
                  t=t+0.2
             else:
                  t=t+0.2
                  s=res[u]
                  speaker.append(s)
                  time.append(t)
        print(time)
        print(speaker)
        speakerN = speaker
        speakerN.append(0)
        for i in range(2, len(time)):
             if((time[i]-time[i-1]) < 0.75):
                  pass
             else:
                  speaker[i-1] = speakerN[i-2]           
        fin = []
        for i in range(1,len(time)):
             if(speaker[i]!=speaker[i-1]):
                  fin.append([time[i-1], speaker[i-1]])
             else:
                  pass
        textOfSound.delete('1.0', END)
        textOfSound.insert(END, "There is no enough information")
        for p in range(0, len(fin)):
             print("TIME : " + str(fin[p][0]) + " ---- " + "SPEAKER : " + str(fin[p][1]))                    
             textOfSound.insert(END, "TIME : " + str(fin[p][0]) + " ---- " + "SPEAKER : " + str(fin[p][1]))  
        plt.style.use("ggplot")
        plt.figure()
        time.append(len(time))
        plt.xlabel("Время")
        plt.ylabel("Говорящий")
        plt.plot(time,speaker)
        plt.show()
        import IPython
        IPython.display.display(IPython.display.Audio(file_name))
    else:
        textOfSound.delete('1.0', END)
        textOfSound.insert(END, "There is no enough information")

# button to diarizate        
btnDiarizate = Button(window, text="Diarizate", fg='blue', height = 5, command = doDiarizate)
btnDiarizate.pack(side=BOTTOM, fill = BOTH)

# title of app
window.title('Voice recognition')
window.minsize(350, 350)
window.geometry("400x400+20+20")
window.mainloop()
