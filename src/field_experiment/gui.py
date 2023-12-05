import PySimpleGUI as sg
import os
sg.set_options(font=('Arial', 20))
sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
phones = ['PixelXL', 'iPhone13', 'Pixel6']
textiles = ['skin', 'cotton', 'polyester', 'thickcotton', 'thickpolyester']
holds = ['vertical']
layout = [ [sg.Text('Smartphone and Stethoscope experiment')],
            [sg.Text('Enter your name as (Xiaoming_Wang)'), sg.InputText(key='name')],
            [sg.Text('Select smartphone'), sg.Listbox(phones, size=(6, 4), key='smartphone'),
              sg.Text('Select textile'), sg.Listbox(textiles, size=(6, 4), key='textile'), 
            sg.Text('Select hold'), sg.Listbox(holds, size=(6, 4), key='hold')],
            [sg.Button('Confirm setting')],
            [sg.Text('Will set output from command line if connections are good')],
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
        name = values['name']
        assert len(name) > 0 
        phone = values['smartphone'][0]
        textile = values['textile'][0]
        hold = values['hold'][0]
        sub_dir = os.path.join('thinklabs', name, '_'.join([phone, textile, hold]))
        print('data will be save to', sub_dir)
        os.makedirs(sub_dir, exist_ok=True)
    elif event == 'Start recording':
        os.system('python adb_collect.py' + ' ' + sub_dir)
        record_num += 1
        window['num'].update('number of recordings: ' + str(record_num))
window.close()