#data_split.py

import os
import librosa
import numpy as np

path_dir = "./data/"
output_dir = "./data_split/"

file_list = os.listdir(path_dir) 

wav_files = [f for f in file_list if f[-12:] == "denoised.wav"]
txt_files = [f for f in file_list if f[-4:] == ".txt"]

wav_files.sort()
txt_files.sort()

print("wav files")
print(wav_files)

print("txt files")
print(txt_files)

data_list_1 = []
data_list_2 = []
data_list_3 = []

for path in txt_files:
    print(path)
    f = open(path_dir+path,'r')
    data =f.read()
    
    data_list = data.split("\n")
    data_list = data_list [:-1]
    data_list = [data for data in data_list if data.find("None",9) == -1]
    
    print(data_list)
    
    temp_hit = [float(data.split(" ")[0]) for data in data_list]
    data_list_1.append(temp_hit)
    data_list_2.append([data.split(" ")[1] for data in data_list])
    data_list_3.append([data.split(" ")[2] for data in data_list])
    
    f.close()
    
#for 1 wav file
for idx1 in range(len(data_list_1)):
    
    y, sr = librosa.load(path_dir+wav_files[idx1],sr=16000)
    
    for idx2 in range(len(data_list_1[idx1])):
        # 시간 계산
        curr_time = data_list_1[idx1][idx2]
        output_path = output_dir + txt_files[idx1][:-9]+data_list_2[idx1][idx2]+"_"+data_list_3[idx1][idx2]+"_"+str(idx1)+"_"+str(idx2)+".wav"
#        print(output_path)
        #Normalize [-1,1]
        temp = y[int(sr*curr_time):int(sr*curr_time)+16384]
        max_amp = np.max(np.abs(temp))
        if max_amp > 1:
            temp /= max_amp
        librosa.output.write_wav(output_path, temp,sr)
