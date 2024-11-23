import pandas as pd
from pandas import read_excel
import numpy as np

file = 'amazon_AI_data.xlsx'
data = read_excel(file)
data.head()
