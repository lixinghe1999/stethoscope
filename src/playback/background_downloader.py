from audioset_download import Downloader
labels = ["Inside, small room", "Inside, large room or hall", "Inside, public space"]
d = Downloader(root_path='evaluation', labels=labels, n_jobs=4, download_type='eval', copy_and_replicate=False)
d.download(format = 'flac')