import PySimpleGUI as sg
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import librosa
def fig_maker(window, sub_dir): # this should be called as a thread, then time.sleep() here would not freeze the GUI
    files = os.listdir(sub_dir)
    if len(files) == 0:
        window.write_event_value('-THREAD-', 'done.')
        return plt.gcf()
    files.sort()
    Mic = [f for f in files if f.split('_')[0] == 'MIC']
    Steth = [f for f in files if f.split('_')[0] == 'Steth']
    mic_recording = librosa.load(os.path.join(sub_dir, Mic[-1]), mono=False, sr=4000)[0]
    steth_recording = librosa.load(os.path.join(sub_dir, Steth[-1]), sr=4000)[0]
    mic_recording = mic_recording / np.max(np.abs(mic_recording), axis=-1, keepdims=True)

    plt.subplot(211)
    plt.plot(mic_recording.T)
    plt.subplot(212)
    plt.plot(steth_recording)
    window.write_event_value('-THREAD-', 'done.')
    return plt.gcf()
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
sg.set_options(font=('Arial', 20))
sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
phones = ['PixelXL', 'iPhone13', 'Pixel6', 'Mate20', 'SAST+EDIFIER']
textiles = ['skin', 'cotton', 'polyester', 'thickcotton', 'thickpolyester', 'PU', 'cowboy']
layout = [ [sg.Text('Smartphone and Stethoscope experiment')],
            [sg.Text('Enter your name as (Xiaoming_Wang)'), sg.InputText(key='name')],
            [sg.Text('Select smartphone'), sg.Listbox(phones, size=(8, 8), key='smartphone'),
              sg.Text('Select textile'), sg.Listbox(textiles, size=(8, 8), key='textile'),
               sg.Canvas(size=(500,500), key='canvas')],
            [sg.Button('Confirm setting'), sg.Button('Start recording'), sg.Button('Plot last')],
            [sg.Text('number of recordings: 0', key='num'), sg.Button('Exit'), ],
            ]

window = sg.Window('Experiment', layout)
record_num = 0
fig_agg = None
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break
    elif event == 'Confirm setting':
        name = values['name']
        assert len(name) > 0 
        phone = values['smartphone'][0]
        textile = values['textile'][0]
        sub_dir = os.path.join('thinklabs', name, '_'.join([phone, textile]))
        print('data will be save to', sub_dir)
        os.makedirs(sub_dir, exist_ok=True)
    elif event == 'Start recording':
        #os.system('python adb_collect.py' + ' ' + sub_dir)
        os.system('python multiprocess_collect.py' + ' ' + sub_dir + ' ' + phone)
        record_num += 1
        window['num'].update('number of recordings: ' + str(record_num))
    elif event == 'Plot last':
        if fig_agg is not None:
                    delete_fig_agg(fig_agg)
        fig = fig_maker(window, sub_dir)
        fig_agg = draw_figure(window['canvas'].TKCanvas, fig)

window.close()