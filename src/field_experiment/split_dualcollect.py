import os
from scipy.io import wavfile
people = [
        'Lixing_He',
         # 'Liangyu_Liu'
         # 'Xuefu_Dong'
         ]
smartphone = ['SAST+EDIFIER']
for p in people:
    sessions = os.listdir(os.path.join('thinklabs', p))
    for s in sessions:
        phone, textile = s.split('_')
        if phone in smartphone:
            s1, s2 = phone.split('+')
            os.makedirs(os.path.join('thinklabs', p, s).replace(phone, s1), exist_ok=True)
            os.makedirs(os.path.join('thinklabs', p, s).replace(phone, s2), exist_ok=True)
            sub_dir = os.path.join('thinklabs', p, s)
            mic = [f for f in os.listdir(sub_dir) if f.split('_')[0] == 'MIC']
            steth = [f for f in os.listdir(sub_dir) if f.split('_')[0] == 'Steth']
            for f1, f2 in zip(mic, steth):
                fs, data = wavfile.read(os.path.join(sub_dir, f1))            # reading the file
                wavfile.write(os.path.join(sub_dir, f1).replace(phone, s1), fs, data[:, 0])   # saving first column which corresponds to channel 1
                wavfile.write(os.path.join(sub_dir, f1).replace(phone, s2), fs, data[:, 1])  

                fs, data = wavfile.read(os.path.join(sub_dir, f2))            # reading the file
                wavfile.write(os.path.join(sub_dir, f2).replace(phone, s1), fs, data)
                wavfile.write(os.path.join(sub_dir, f2).replace(phone, s2), fs, data)
