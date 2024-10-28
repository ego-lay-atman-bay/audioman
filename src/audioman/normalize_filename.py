import os
import pathvalidate

import logging

def normalize_filename(filename: str) -> str:
    filename = os.path.abspath(filename)
    filename = filename[0:3] + filename[3:].replace(':', '').replace("?", "")
    
    filename = pathvalidate.sanitize_filepath(filename, replacement_text = '-')

    logging.debug(f'filename: {filename}')

    return filename
