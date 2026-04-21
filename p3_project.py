import sys
import os
import csv
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sk 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def remove_class_and_scale(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Category'])
    df = df.drop(columns=['Sex'])
    df = df.drop(columns=['Age'])

    # replace missing values with the mean of each column
    df = df.fillna(df.mean())
    
    # scale using standard scaler

    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(df)

    # convert the scaled data back to a DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=[f'PC{i+1}' for i in range(X_scaled.shape[1])])
    
    # save as a new CSV file
    X_scaled.to_csv("all_data_csvs/data_scaled.csv", index=False)

def plot_pca(file_path, n_components=2):
    df = pd.read_csv(file_path)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df)

    if n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('PCA with 3 Components')
        plt.show()
    elif n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA with 2 Components')
        plt.show()
    elif n_components == 1:
        plt.figure(figsize=(8, 6))
        plt.hist(X_pca[:, 0], bins=30, edgecolor='k')
        plt.xlabel('Principal Component 1')
        plt.title('PCA with 1 Component')
        plt.show()
    else:
        print("bruh")

def plot_mds(file_path, n_components=2):
    df = pd.read_csv(file_path)
    mds = MDS(n_components=n_components)
    X_mds = mds.fit_transform(df)

    if n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_mds[:, 0], X_mds[:, 1], X_mds[:, 2])
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        ax.set_zlabel('MDS Component 3')
        plt.title('MDS with 3 Components')
        plt.show()
    elif n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_mds[:, 0], X_mds[:, 1])
        plt.xlabel('MDS Component 1')
        plt.ylabel('MDS Component 2')
        plt.title('MDS with 2 Components')
        plt.show()
    elif n_components == 1:
        plt.figure(figsize=(8, 6))
        plt.hist(X_mds[:, 0], bins=30, edgecolor='k')
        plt.xlabel('MDS Component 1')
        plt.title('MDS with 1 Component')
        plt.show()
    else:
        print("bruh")

if __name__ == "__main__":
    
    file_path = "all_data_csvs/data_scaled.csv"
    plot_mds(file_path, n_components=3)
    plot_mds(file_path, n_components=2)
    plot_mds(file_path, n_components=1)
    
