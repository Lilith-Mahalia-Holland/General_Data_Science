import PyPDF2
import pandas as pd
import re
import os
import sys
import string
from collections import Counter

base = "F:/Machine Learning/PDF's"
if not os.path.exists(base):
    sys.exit()

file_path = []
for path in os.walk(base):
    if len(path[1]) == 0:
        root, _, files = path
        for file in files:
            file_path.append(os.path.join(root, file))