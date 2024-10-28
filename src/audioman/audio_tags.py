import io
import os
import re
import typing
from typing import Any, overload
from copy import deepcopy

import mutagen
import PIL
from mutagen import id3
from PIL import Image

from . import _tags
from .normalize_filename import normalize_filename


class AudioTags(dict):
    def __init__(self, file: str | dict | None = None) -> None:
        """Audio metadata tags. This serves as a way to set audio metadata on different file formats.

        Args:
            file (str, optional): File to import metadata tags from. This can be file path, or filelike object. Defaults to None.
        """
        self.filename = None
        self._picture: Image.Image = None
        self.picture_filename: str | None = None
        
        if isinstance(file, str) or file == None:
            super().__init__()
            
            self.load(file)
        else:
            super().__init__(file)

        
    def load(self, file: str = None):
        """Load file.

        Args:
            file (str, optional): File can be path to file, or file object. Defaults to `self.filename`.
        """
        if file == None:
            file = self.filename
        
        self.clear()
        
        self.filename = None
        
        if file:
            audio = mutagen.File(file)
            if audio == None:
                return
            
            self.filename = audio.filename
            if isinstance(audio.tags, id3.ID3):
                audio.tags.update_to_v24()

            keys = audio.keys()

            tags = [_tags.get_tag_name(key) for key in keys]

            for tag in tags:
                if tag == 'picture':
                    continue
                
                self.__setitem__(
                    tag,
                    _tags.get_tag(audio, tag),
                )
            
            self.picture = _tags.get_picture(audio)

    def save(self, file: str = None):
        """Save tags to file.

        Args:
            file (str, optional): File to save tags to. Can be file path or filelike object. Defaults to `self.filename`.

        Raises:
            AttributeError: must provide file
        """
        if file == None and self.filename != None:
            file = self.filename
        else:
            if file == None:
                raise AttributeError('must provide file')
        
        audio: mutagen.FileType = mutagen.File(file)
        if isinstance(audio.tags, id3.ID3):
            audio.tags.update_to_v24()

        if audio.tags == None:
            audio.add_tags()

        audio.tags.clear()

        for tag in self:
            _tags.set_tag(audio, tag, self[tag])

        _tags.set_picture(audio, self.picture)

        try:
            audio.save(file, v2_version = 4)
        except:
            audio.save(file)
            

    def __setitem__(self, key: str, value: str | list) -> None:
        if not isinstance(key, str):
            raise TypeError('key must be str')

        key = key.lower()
        key = _tags.get_tag_name(key)

        if key == 'picture':
            self.picture = value
            return
        
        if isinstance(value, str) and ';' in value:
            value = [str(p).replace(r'\;', ';') for p in re.split(r'(?<!\\);', value)]
            if len(value) == 0:
                value = None
            elif len(value) == 1:
                value = value[0]

        return super().__setitem__(key, value)
    
    def set(self, tag: str, value: str | list) -> None:
        """Set tag value.

        Args:
            tag (str): Tag name.
            value (str | list): Tag value.
        """
        return self.__setitem__(tag, value)

    def __getitem__(self, key: str) -> str | list:
        if not isinstance(key, str):
            raise TypeError('key must be str')

        key = key.lower()
        key = _tags.get_tag_name(key)

        return super().__getitem__(key)

    def get(self, tag: str, default: typing.Any = None):
        """Get tag value.

        Args:
            tag (str): Tag name.
            default (Any, optional): Default value if tag does not exist. Defaults to None.

        Raises:
            TypeError: tag must be str

        Returns:
            str | list[str]: tag value.
        """
        if not isinstance(tag, str):
            raise TypeError('tag must be str')

        tag = tag.lower()
        tag = _tags.get_tag_name(tag)

        return super().get(tag, default)

    def __contains__(self, key: str) -> bool:
        if not isinstance(key, str):
            raise TypeError('key must be str')

        key = key.lower()
        key = _tags.get_tag_name(key)

        return super().__contains__(key)

    def pop(self, tag: str, default: typing.Any):
        if not isinstance(tag, str):
            raise TypeError('tag must be str')

        tag = tag.lower()
        tag = _tags.get_tag_name(tag)

        return super().pop(tag, default)

    @overload
    def setdefault(self, tag: str, default: str = None) -> Any:
        """If the tag exists, don't change it, if it doesn't exit, add it with the default value.

        Args:
            tag (str): Tag
            default (str, optional): Default value. Defaults to None.

        Raises:
            TypeError: Tag must be str

        Returns:
            Any: The value of the 
        """
        ...
    @overload
    def setdefault(self, tag: dict[str, str]) -> None:
        """Set multiple tags and their defaults.

        Args:
            tag (dict[str, str]): Dictionary of tags and defaults
        """
        ...
    def setdefault(self, tag: str, default: str = None):
        if not isinstance(tag, (str, dict, list)):
            raise TypeError('tag must be str')
        
        if isinstance(tag, dict):
            for name in tag:
                self.setdefault(name, tag[name])
            return
        elif isinstance(tag, list):
            for name in tag:
                self.setdefault(name[0], name[1])
            return

        tag = tag.lower()
        tag = _tags.get_tag_name(tag)
        
        if tag not in self:
            self.__setitem__(tag, default)
        
        return self.get(tag)

    
    def update(self, values: dict | typing.Iterable[tuple], **kwargs):
        if isinstance(values, dict):
            for key in values:
                self.__setitem__(key, values[key])
        elif isinstance(values, list):
            for item in values:
                self.__setitem__(item[0], item[1])
        for key in kwargs:
            self.__setitem__(key, kwargs[key])
    
    def expand(self):
        """Expand tags to include all available tag names. This creates a new dict with all the values, not an AudioTags object.

        Returns:
            dict[str,str | list[str]]: Tags.
        """
        tags = dict()
        for tag in self:
            for name in _tags.get_tag_names(tag):
                tags[name] = self[tag]
        
        return tags

    @property
    def picture(self) -> Image.Image | None:
        """Audio picture.

        Returns:
            PIL.Image | None: PIL Image
        
        Set:
            image (PIL.Image | str | bytes | filelike | None): Image to set to. Can be PIL Image, path to file, file bytes, filelike object, or None.
        """
        return self._picture
    @picture.setter
    def picture(self, image: Image.Image | str | bytes | io.BytesIO | None):
        try:
            picture = None
            if image == None:
                picture = None
            elif isinstance(image, str) or (hasattr(image, 'read') and hasattr(image, 'seek') and hasattr(image, 'tell')):
                if isinstance(image, str):
                    image = normalize_filename(image)
                if os.path.exists(image):
                    self.picture_filename = image
                    picture = Image.open(image)
            elif isinstance(image, bytes):
                image = io.BytesIO(image)
                image.seek(0)
                picture = Image.open(image)
            elif isinstance(image, Image.Image):
                picture = image.copy()
            else:
                picture = None
        except PIL.UnidentifiedImageError:
            picture = None

        self._picture = picture
    
    def get_tag_name(self, tag):
        return _tags.get_tag_name(tag)
    
    def copy(self) -> 'AudioTags':
        """A deep copy of AudioTags.

        Returns:
            AudioTags: Copy of AudioTags.
        """
        return deepcopy(self)
