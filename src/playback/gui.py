import PySimpleGUI as sg
import os
import numpy as np
from multiprocess_collect import dual_record
sg.set_options(font=('Arial', 20))
sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
phones = ['PixelXL', 'iPhone13', 'Pixel6', 'ThinkLabs']
textiles = ['nothing', 'cotton', 'polyester', 'PU', 'cowboy']
layout = [ [sg.Text('Smartphone and Playback experiment')],
            [sg.Text('Enter playback dataset (heartbeat)'), sg.InputText(key='dataset')],
            [sg.Text('Select smartphone'), sg.Listbox(phones, size=(8, 6), key='smartphone'),
              sg.Text('Select textile'), sg.Listbox(textiles, size=(8, 6), key='textile')],
            [sg.Button('Confirm setting')],
            [sg.Button('Start recording')],
            [sg.Text('number of recordings: 0', key='num')],
            [sg.Button('Exit')]
            ]

window = sg.Window('Experiment', layout)
record_num = 0
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break
    elif event == 'Confirm setting':
        data = values['dataset']
        if len(data) == 0:
            data = 'heartbeat'
        phone = values['smartphone'][0]
        textile = values['textile'][0]
        print(data, phone, textile)
    elif event == 'Start recording':
        play_file = os.path.join('dataset', data+'.wav')
        save_file = os.path.join('dataset', '_'.join([data, phone, textile])+'.wav')
        print('playback file:', play_file)
        # dual_record(play_file)
        os.system('python multiprocess_collect.py' + ' ' + play_file + ' ' + save_file)
        # os.system('python adb_collect.py' + ' ' + play_file)
        record_num += 1
        window['num'].update('number of recordings: ' + str(record_num))
window.close()