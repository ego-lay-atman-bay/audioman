import os
import logging
import subprocess
import charset_normalizer
import tempfile


ffmpeg_log_filename = 'ffreport.log'

def setup_ffmpeg_log(filename = ffmpeg_log_filename, level = 32):
    """Setup ffmpeg logging.

    Args:
        filename (str, optional): Output log. Defaults to 'ffreport.log'.
        level (int, optional): Log level. Levels are defined at https://ffmpeg.org/ffmpeg.html#toc-Generic-options. Defaults to 32 (info).
    """
    os.environ['FFREPORT'] = f"file='{filename}':level={level}"

def log_ffmpeg_output(filename = ffmpeg_log_filename, level: int = logging.INFO, delete: bool = True):
    """Log the ffmpeg log using the python `logging` module.

    Args:
        filename (str, optional): Output log. Defaults to 'ffreport.log'.
        level (int, optional): Log level. Defaults to `logging.INFO`.
        delete (bool, optional): Delete the log file after logging it. Defaults to True.
    """
    file = charset_normalizer.from_path(filename).best()
    logging.log(msg = f'ffmpeg output\n\n{file.output().decode()}\n', level = level)
    
    if delete:
        if os.path.exists(filename):
            os.remove(filename)

def run_ffmpeg_command(ffmpeg_options: list[str] | str, log_level = logging.DEBUG):
    test_ffmpeg()
    
    command = ['ffmpeg']
    
    if isinstance(ffmpeg_options, (list, tuple)):
        command += ffmpeg_options
    else:
        command = f"{' '.join(command)} {ffmpeg_options}"
    
    
    logging.debug(f'command:\n{command}')
    
    with tempfile.NamedTemporaryFile(
            'w',
            prefix = 'audioman_',
            suffix = '.log',
            delete = False,
        ) as file:
            file.write('')
    
    setup_ffmpeg_log(file.name)
    result = subprocess.run(command)
    log_ffmpeg_output(file.name, log_level)
    
    result.check_returncode()

def test_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error'])
    except:
        raise FileNotFoundError('Cannot run ffmpeg. Make sure ffmpeg is available on the PATH.')
