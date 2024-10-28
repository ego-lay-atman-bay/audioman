import re

class FormattedList(list):
    def __format__(self, format_spec: str) -> str:
        separator = ', '
        if format_spec:
            separator = format_spec
        return separator.join(self)

class EscapeFormat():
    def __init__(self, key):
        if isinstance(key, (list, tuple, set)):
            key = FormattedList(key)
        self.key = key

    def __format__(self, spec: str):
        result = self.key
        if spec.lower() == "fftext":
            result = str(result)
            escape_chars = r'\:"()[]'
            result = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', result)
            result = result.replace("'", r"'\\\''")
        else:
            result = self.key.__format__(spec)
        return result
    
    def __str__(self) -> str:
        return str(self.key)
    
    def __repr__(self) -> str:
        return repr(self.key)

class SafeFormatDict(dict):
    def __missing__(self, key):
        return f'{{{key}}}'
    
    def __getitem__(self, key):
        return EscapeFormat(super().__getitem__(key))

def format_str(string: str, **values: dict[str,str]):
    for key in values:
        try:
            values[key] = int(values[key])
        except:
            try:
                values[key] = float(values[key])
            except:
                pass
    return string.format_map(SafeFormatDict(values))
