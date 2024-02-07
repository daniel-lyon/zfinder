import re

with open('./zfinder/__init__.py', 'r') as f:
    contents = f.read()
    VERSION = re.search(r"__version__ = '(.+)'", contents).group(1)
    
print(VERSION)