# Importation des bibliotheques

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Importation des donees
try :
    data = pd.read_csv(r'D:\Documents\self learning\ML\Student Depression\Student_depression_analysis\data\raw\student_depression_dataset.csv')
    print(data.head())

except FileNotFoundError:
    print("File not found")

except Exception as e:
    print(f"An error occurred: {e}")

# Data cleaning

