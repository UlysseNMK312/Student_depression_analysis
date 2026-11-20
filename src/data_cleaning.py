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

# -------------------------------
#  Standardisation des noms de colonnes

def standardize_column_names(df):
    """ Standardizes the column names of the DataFrame by stripping whitespace, 
    converting to lowercase, and replacing spaces with underscores."""

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# --------------------------------------------
# Suppression des doublons

def remove_duplicates(df):
    """ Removes duplicate rows from the DataFrame."""
    df = df.drop_duplicates()
    return df

# --------------------------------------------
# Correcting data types

def correct_data_types(df):
    """
    Corrects the data types of the columns in the DataFrame."""

    for col in df.columns:
        if col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip().str.lower()

        if col in df.select_dtypes(include=['int64']):
            df[col] = df[col].astype(int)
    
    return df

# --------------------------------------------
# Outliers detection and removal

def detect_outliers(df, column):
    """ Detects outliers in a specified column of the DataFrame using the IQR method."""

    if column is None or column not in df.columns:
        raise ValueError("Column name must be provided and must exist in the DataFrame.")

    for col in column:
        if col not in df.columns:
            continue
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"\ncolumn: {column}")
    print(f"Number of outliers in {column}: {len(outliers)}")
    
    return outliers

# -------------------------------------------------
# Outliers removal

def remove_outliers(df, column):
    """ Removes outliers from a specified column of the DataFrame using the IQR method."""

    df_cleaned = df.copy()

    for column in data.select_dtypes(include=[np.number]).columns:
        outliers = detect_outliers(data, column)

        if not outliers.empty:
            print(f"Outliers detected in column '{column}':")
            #print(outliers)
            df_cleaned = df[~df.index.isin(outliers.index)]
            print(f"Removed {len(outliers)} outliers from column '{column}'.")

        else:
            print(f"No outliers detected in column '{column}'.")
        
    return df_cleaned

# ----------------------------------------------
# Validation des donneee

def validate_data(df, rules):
    """ Validates the DataFrame by checking for missing values and duplicates."""

    if rules:
        for col, (min, max) in rules.items():
            if col in df.columns:
                df = df[
                    (df[col] >= min) & (df[col] <= max)
                ]

    print("\nData validation:")
    print(f"Dimensions finales : {df.shape}")

    return df

# -------------------------------------------
# Sauvegarde des donnees

def sauvegarder_donnees(df, path):
    """
        Sauvegarde du jeu de donnees ainsi nettoye
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nDonnees netoyees enregistrees dans : {path}")



data = standardize_column_names(data)
data = remove_duplicates(data)
data = correct_data_types(data)
removed_outliers_data = remove_outliers(data, data.select_dtypes(include=[np.number]).columns)
print(f"\n\n\nData shape after removing outliers: {removed_outliers_data.shape}")

# rules dictionnary
rules = {
    "age" : (18, 35),
    "academic_pressure" : (0, 10),
    "work_pressure" : (0, 5),
    "cgpa" : (0, 10),
    "study_satisfaction" : (0, 5),
    "job_satisfaction" : (0, 5),
    "work/study_hours" : (0, 12),
    "financial_stress" : (0, 5),
    "depression" : (0, 1)
}


data = validate_data(data, rules)
print(data)

path = r"D:\Documents\self learning\ML\Student Depression\Student_depression_analysis\data\raw\student_depression_dataset.csv"
sauvegarder_donnees(data, path)
