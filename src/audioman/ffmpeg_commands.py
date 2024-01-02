import os
import logging

ffmpeg_log_filename = 'ffreport.log'

def setup_ffmpeg_log(filename = ffmpeg_log_filename, level = 32):
    """Setup ffmpeg logging.

    Args:
        filename (str, optional): Output log. Defaults to 'ffreport.log'.
        level (int, optional): Log level. Levels are defined at https://ffmpeg.org/ffmpeg.html#toc-Generic-options. Defaults to 32 (info).
    """
    os.environ['FFREPORT'] = f'file={filename}:level={level}'

def log_ffmpeg_output(filename = ffmpeg_log_filename, level: int = logging.INFO, delete: bool = True):
    """Log the ffmpeg log using the python `logging` module.

    Args:
        filename (str, optional): Output log. Defaults to 'ffreport.log'.
        level (int, optional): Log level. Defaults to `logging.INFO`.
        delete (bool, optional): Delete the log file after logging it. Defaults to True.
    """
    with open(ffmpeg_log_filename, 'r') as file:
        logging.log(msg = f'ffmpeg output\n\n{file.read()}\n', level = level)
    
    if delete:
        if os.path.exists(filename):
            os.remove(filename)
