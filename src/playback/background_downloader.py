from audioset_download import Downloader
labels = ["Door", "Cupboard open or close", "Drawer open or close", "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", "Frying (food)", 
          "Microwave oven", "Blender", "Kettle whistle", "Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer",
          "Toilet flush", "Toothbrush", "Vacuum cleaner", "Zipper (clothing)", "Velcro, hook and loop fastener", "Keys jangling", "Coin (dropping)",
          "Packing tape, duct tape", "Scissors", "Electric shaver, electric razor", "Shuffling cards", "Typing", "Writing"]
d = Downloader(root_path='evaluation', labels=labels, n_jobs=4,
                download_type='Evaluation', copy_and_replicate=False)
d.download(format = 'vorbis')