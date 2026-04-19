import sys
import os
import csv
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit # type: ignore
from sklearn.preprocessing import OrdinalEncoder # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from ucimlrepo import fetch_ucirepo # type: ignore

def fetch_data_ucimlrepo(dataset_id=int, save_csv=bool):
    # use to fetch from the repo
    # dataset id for HCV dataset is 571

    # fetch dataset 
    hcv_data = fetch_ucirepo(id=dataset_id) 
    
    # data (as pandas dataframes) 
    X = hcv_data.data.features 
    y = hcv_data.data.targets 
    
    # metadata 
    print(hcv_data.metadata) 
    
    # variable information 
    print(hcv_data.variables)

    # combign features and targets into one dataframe
    df = pd.concat([y, X], axis=1)

    # print first 5 rows of data to check
    print(df.head())

    if save_csv == True:
        csv_file_name = input ("name of csv file: ") + ".csv"
        df.to_csv(csv_file_name, index=False)
        print(f"CSV file '{csv_file_name}' created successfully.")
    else:
        pass

"""
Notes on Data:
plots that work well:

"AST" vs "CHE"
"BIL" vs "CREA"
"AST" vs "BIL"


"""

class data_hsv:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def clean_data(self):
        df = pd.read_csv(self.file_path)

        df_cleaned = df.dropna()

        df_cleaned['Sex'] = df_cleaned['Sex'].replace({'m':1 ,'f':0})

        col_names = df_cleaned.columns[3:]
        features = df_cleaned[col_names]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        df_cleaned[col_names] = features_scaled

        csv_new_path = f"{self.file_path.split('.')[0]}_cleaned.csv"

        with open(csv_new_path, mode='w', newline='') as file:
            df_cleaned.to_csv(file, index=False)
            
        print(f"CSV file '{csv_new_path}' created successfully.")
        
    def graph_data_all(self):
        df = pd.read_csv(self.file_path)
        diagnosis = df.columns[0]
        col_names = df.columns[3:]
        
        plt.show()

    def graph_data_xy(self, x_data, y_data):
        df = pd.read_csv(self.file_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_data, y=y_data, hue='Category')
        plt.title(f'Scatter Plot of {x_data} vs {y_data}')
        plt.show()

    def graph_data_hist(self, feature):
        df = pd.read_csv(self.file_path)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='Category', kde=True)
        plt.title(f'Histogram of {feature}')
        plt.show()

    def data_mean(self):
        df = pd.read_csv(self.file_path)
        col_names = df.columns[3:]
        features = df[col_names]
        means = features.mean()
        print("Mean values for each feature:")
        print(means)

    def correlation_covariance_matrix(self):
        df = pd.read_csv(self.file_path)
        col_names = df.columns[3:]
        features = df[col_names]
        
        correlation_matrix = features.corr()

        covariance_matrix = features.cov()

        print("Correlation Matrix:")
        print(correlation_matrix)

        print("Covariance Matrix:")
        print(covariance_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Covariance Heatmap') 
        plt.show()

    def print_split_info(self):
        file_path = self.file_path
        df = pd.read_csv(file_path)

        # Create the splitting object
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

        # Apply the split to the data frame using the "diagnosis" column as our label
        for train_index, test_index in split.split(df, df["diagnosis"]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        #print
        print('Overall class balance:')
        print('{}'.format(df["diagnosis"].value_counts() / len(df)))
        print('Train set class ratio:')
        print('{}'.format(train_set["diagnosis"].value_counts() / len(train_set)))
        print('Test set class ratio:')
        print('{}'.format(test_set["diagnosis"].value_counts() / len(test_set)))

        
        df = pd.read_csv(file_path)

        # Create the splitting object
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

        # Apply the split to the data frame using the "diagnosis" column as our label
        for train_index, test_index in split.split(df, df["diagnosis"]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        #print
        print('Overall class balance:')
        print('{}'.format(df["diagnosis"].value_counts() / len(df)))
        print('Train set class ratio:')
        print('{}'.format(train_set["diagnosis"].value_counts() / len(train_set)))
        print('Test set class ratio:')
        print('{}'.format(test_set["diagnosis"].value_counts() / len(test_set)))



if __name__ == "__main__":
    
    data = data_hsv("data_project_2.csv")
    data.clean_data()

    print ("\n\ndone")

