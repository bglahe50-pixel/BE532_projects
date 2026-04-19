import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import re
import os
from PIL import Image # type: ignore


def graph_data_histogram(df, col, save_png=False):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, hue='Category', kde=True)
    plt.title(f'Histogram of {col}')

    if save_png:
        plt.savefig(f"histogram_{col}.png")
    plt.show()


def graph_data_scatter(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='Category')
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.show()


def id_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers


def condense_cms(directory, regex):
    # puts all cms side by side into one png file
    images = []
    for filename in os.listdir(directory):
        if re.match(regex, filename):
            img = Image.open(os.path.join(directory, filename))
            images.append(img)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    new_img.save(os.path.join(directory, "condensed_cms.png"))


def identify_outliers(df):
    # return a csv file with each feature value set to either "outlier" or "not outlier"
    outlier_df = pd.DataFrame()
    for col in df.columns[2:]:
        outliers = id_outliers(df, col)
        outlier_df[col] = df[col].apply(lambda x: 1 if x in outliers[col].values else 0)
    outlier_df.to_csv("outliers.csv", index=True)
        
if __name__ == "__main__":
    # target_file = "all_data_csvs/data_scaled.csv"
    # df = pd.read_csv(target_file)
    identify_outliers(pd.read_csv("all_data_csvs/data_scaled.csv"))

    print("\n\nDone")