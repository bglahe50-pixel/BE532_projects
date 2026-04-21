import sys
import os
import csv
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import sklearn as sk # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from ucimlrepo import fetch_ucirepo # type: ignore

def fetch_data_ucimlrepo(dataset_id:int, save_csv:bool):
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
        csv_file_name = "data_raw.csv"
        df.to_csv(csv_file_name, index=False)
    else:
        pass


def clean_data(file_path):
    df = pd.read_csv(file_path)

    # remove sex
    df = df.drop(columns=['Sex'])

    # convert all non blood-donor classes to one sick class
    df['Category'] = df['Category'].replace({
        "0=Blood Donor":0, 
        "0s=suspect Blood Donor":1, 
        "1=Hepatitis":1,
        "2=Fibrosis":1,
        "3=Cirrhosis":1})

    # remove all columns that have a missing value
    df = df.dropna(axis=1)

    csv_new_path = f"{file_path.split('_')[0]}_cleaned.csv"
        
    """
    some people were missing: 
    - ALB
    - ALP
    - CHOL
    - PROT
    - ALT
    """

    with open(csv_new_path, mode='w', newline='') as file:
        df.to_csv(file, index=False)


def scale_data(file_path, scaler):
    df = pd.read_csv(file_path)

    # scaler options:
    # sk.preprocessing.MinMaxScaler
    # sk.preprocessing.OrdinalEncoder
    # sk.preprocessing.StandardScaler

    #we're using standard scaler for this project because our dataset has some outliers 
    
    # standardize features
    # but not index or class
    col_names = df.columns[2:]
    features = df[col_names]
    features_scaled = scaler.fit_transform(features)
    df[col_names] = features_scaled
    
    csv_new_path = f"{file_path.split('_')[0]}_scaled.csv"

    with open(csv_new_path, mode='w', newline='') as file:
        df.to_csv(file, index=False)

    
def stratified_shuffle_split(df, target_col = 'Category', test_size = 0.2, random_state = 42):
    split = sk.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(df, df[target_col]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # make a directory to save the csv files if it doesn't exist
    if not os.path.exists(f"rs_{random_state}"):
        os.makedirs(f"rs_{random_state}")

    # save the stratified train and test sets to new csv files
    strat_train_set.to_csv(f"rs_{random_state}/strat_train_set-{random_state}.csv", index=False)
    strat_test_set.to_csv(f"rs_{random_state}/strat_test_set-{random_state}.csv", index=False)

    return strat_train_set, strat_test_set


class data_state:
    def __init__(self, random_state):
        self.random_state = random_state
        self.train_file = f"rs_{random_state}/strat_train_set-{random_state}.csv"
        self.test_file = f"rs_{random_state}/strat_test_set-{random_state}.csv"
        self.target_dir = f"rs_{random_state}/"


    def graph_data_histogram(self, feature, save_png = False):
        df = pd.read_csv(self.train_file)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='Category', kde=True)
        plt.title(f'Histogram of {feature} by Category')
        if save_png:
            plt.savefig(f"{self.target_dir}histogram_{feature}.png")
        plt.show()


    def graph_data_scatter(self, x_data, y_data, save_png = False):
        df = pd.read_csv(self.train_file)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_data, y=y_data, hue='Category')
        plt.title(f'Scatter Plot of {x_data} vs {y_data}')
        if save_png:
            plt.savefig(f"{self.target_dir}scatter_{x_data}_vs_{y_data}.png")
        plt.show()




    def plot_confusion_matrix(self, y_true, y_pred, title, save_png = False):
        cm = sk.metrics.confusion_matrix(y_true, y_pred)
        disp = sk.metrics.ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        if save_png:
            plt.savefig(f"{self.target_dir}{title}.png")
        plt.close()


    def area_under_roc_curve(self, y_true, y_pred_prob, title, save_png = False):

        auc = sk.metrics.roc_auc_score(y_true, y_pred_prob)

        # plot using matplotlib
        fpr, tpr, _ = sk.metrics.roc_curve(y_true, y_pred_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title}')
        plt.legend(loc="lower right")
        if save_png:
            plt.savefig(f"{self.target_dir}ROC {title}.png")
        plt.close()


    def identify_misclassified(self, y_true, y_pred):
        misclassified_indices = np.where(y_true != y_pred)[0]
        return misclassified_indices  


    def identify_outliers_total(self):
        df = pd.read_csv(self.test_file)
        col_names = df.columns[2:]
        for feature in col_names:  
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            print(f"Outliers in {feature}:")
            print(outliers[[feature, 'Category']])
        

    def native_bayes_total(self):
        df = pd.read_csv(self.train_file)

        gnb = sk.naive_bayes.GaussianNB()
        X_train = df.drop(columns=['Category'])
        y_train = df['Category']
        gnb.fit(X_train, y_train)

        y_pred_train = gnb.predict(X_train)

        # test model against test set
        df_test = pd.read_csv(self.test_file)
        X_test = df_test.drop(columns=['Category'])
        y_test = df_test['Category']
        y_pred_test = gnb.predict(X_test)

        y_pred_test_prob = gnb.predict_proba(X_test)[:, 1]

        return y_train, y_pred_train, y_test, y_pred_test, y_pred_test_prob


    def native_bayes_specific(self, feature, in_ex = "include"):
        if in_ex == "include":
            df = pd.read_csv(self.train_file)

            gnb = sk.naive_bayes.GaussianNB()
            X_train = df[[feature]]
            y_train = df['Category']
            gnb.fit(X_train, y_train)

            y_pred_train = gnb.predict(X_train)

            # test model against test set
            df_test = pd.read_csv(self.test_file)
            X_test = df_test[[feature]]
            y_test = df_test['Category']
            y_pred_test = gnb.predict(X_test)

            y_pred_test_prob = gnb.predict_proba(X_test)[:, 1]

        elif in_ex == "exclude":
            df = pd.read_csv(self.train_file)

            gnb = sk.naive_bayes.GaussianNB()
            X_train = df.drop(columns=['Category', feature])
            y_train = df['Category']
            gnb.fit(X_train, y_train)

            y_pred_train = gnb.predict(X_train)

            # test model against test set
            df_test = pd.read_csv(self.test_file)
            X_test = df_test.drop(columns=['Category', feature])
            y_test = df_test['Category']
            y_pred_test = gnb.predict(X_test)

            y_pred_test_prob = gnb.predict_proba(X_test)[:, 1]

        else:
            raise ValueError("in_ex parameter must be either 'include' or 'exclude'")

        return y_train, y_pred_train, y_test, y_pred_test, y_pred_test_prob


    def mr_rodgers_total(self, n_neighbors = 5):
        df = pd.read_csv(self.train_file)

        knn = sk.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors)
        X_train = df.drop(columns=['Category'])
        y_train = df['Category']
        knn_done = knn.fit(X_train, y_train)

        y_pred = knn_done.predict(X_train)

        df_test = pd.read_csv(self.test_file)
        X_test = df_test.drop(columns=['Category'])
        y_test = df_test['Category']
        y_pred_test = knn_done.predict(X_test)

        y_pred_test_prob = knn_done.predict_proba(X_test)[:, 1]

        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


    def mr_rodgers_specific(self, feature, in_ex = "include", n_neighbors = 5):
        if in_ex == "include":
            df = pd.read_csv(self.train_file)

            knn = sk.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors)
            X_train = df[[feature]]
            y_train = df['Category']
            knn_done = knn.fit(X_train, y_train)

            y_pred = knn_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test[[feature]]
            y_test = df_test['Category']
            y_pred_test = knn_done.predict(X_test)

            y_pred_test_prob = knn_done.predict_proba(X_test)[:, 1]

        elif in_ex == "exclude":
            df = pd.read_csv(self.train_file)

            knn = sk.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors)
            X_train = df.drop(columns=['Category', feature])
            y_train = df['Category']
            knn_done = knn.fit(X_train, y_train)

            y_pred = knn_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test.drop(columns=['Category', feature])
            y_test = df_test['Category']
            y_pred_test = knn_done.predict(X_test)

            y_pred_test_prob = knn_done.predict_proba(X_test)[:, 1]

        else:
            raise ValueError("in_ex parameter must be either 'include' or 'exclude'")

        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


    def support_vector_machine_total(self, kernel = 'rbf'):
        df = pd.read_csv(self.train_file)

        svm = sk.svm.SVC(kernel=kernel, probability=True)
        X_train = df.drop(columns=['Category'])
        y_train = df['Category']
        svm_done = svm.fit(X_train, y_train)

        y_pred = svm_done.predict(X_train)

        df_test = pd.read_csv(self.test_file)
        X_test = df_test.drop(columns=['Category'])
        y_test = df_test['Category']
        y_pred_test = svm_done.predict(X_test)

        y_pred_test_prob = svm_done.predict_proba(X_test)[:, 1]

        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


    def support_vector_machine_specific(self, feature, in_ex = "include", kernel = 'rbf'):
        if in_ex == "include":
            df = pd.read_csv(self.train_file)

            svm = sk.svm.SVC(kernel=kernel, probability=True)
            X_train = df[[feature]]
            y_train = df['Category']
            svm_done = svm.fit(X_train, y_train)

            y_pred = svm_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test[[feature]]
            y_test = df_test['Category']
            y_pred_test = svm_done.predict(X_test)

            y_pred_test_prob = svm_done.predict_proba(X_test)[:, 1]

        elif in_ex == "exclude":
            df = pd.read_csv(self.train_file)

            svm = sk.svm.SVC(kernel=kernel, probability=True)
            X_train = df.drop(columns=['Category', feature])
            y_train = df['Category']
            svm_done = svm.fit(X_train, y_train)

            y_pred = svm_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test.drop(columns=['Category', feature])
            y_test = df_test['Category']
            y_pred_test = svm_done.predict(X_test)

            y_pred_test_prob = svm_done.predict_proba(X_test)[:, 1]

        else:
            raise ValueError("in_ex parameter must be either 'include' or 'exclude'")

        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


    def decision_tree_total(self, criterion = 'gini'):
        df = pd.read_csv(self.train_file)

        dt = sk.tree.DecisionTreeClassifier(criterion=criterion)
        X_train = df.drop(columns=['Category'])
        y_train = df['Category']
        dt_done = dt.fit(X_train, y_train)

        y_pred = dt_done.predict(X_train)

        df_test = pd.read_csv(self.test_file)
        X_test = df_test.drop(columns=['Category'])
        y_test = df_test['Category']
        y_pred_test = dt_done.predict(X_test)

        y_pred_test_prob = dt_done.predict_proba(X_test)[:, 1]

        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


    def decision_tree_specific(self, feature, in_ex = "include", criterion = 'gini'):
        if in_ex == "include":
            df = pd.read_csv(self.train_file)

            dt = sk.tree.DecisionTreeClassifier(criterion=criterion)
            X_train = df[[feature]]
            y_train = df['Category']
            dt_done = dt.fit(X_train, y_train)

            y_pred = dt_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test[[feature]]
            y_test = df_test['Category']
            y_pred_test = dt_done.predict(X_test)

            y_pred_test_prob = dt_done.predict_proba(X_test)[:, 1]

        elif in_ex == "exclude":
            df = pd.read_csv(self.train_file)

            dt = sk.tree.DecisionTreeClassifier(criterion=criterion)
            X_train = df.drop(columns=['Category', feature])
            y_train = df['Category']
            dt_done = dt.fit(X_train, y_train)

            y_pred = dt_done.predict(X_train)

            df_test = pd.read_csv(self.test_file)
            X_test = df_test.drop(columns=['Category', feature])
            y_test = df_test['Category']
            y_pred_test = dt_done.predict(X_test)

            y_pred_test_prob = dt_done.predict_proba(X_test)[:, 1]

        else:
            raise ValueError("in_ex parameter must be either 'include' or 'exclude'")
        
        return y_train, y_pred, y_test, y_pred_test, y_pred_test_prob


if __name__ == "__main__":
    random_state = [21, 42, 67, 68, 100]
    for random_state in random_state:
        file_path = f"rs_{random_state}/strat_train_set-{random_state}.csv"
        df = pd.read_csv(file_path)
        file_path = f"rs_{random_state}/strat_train_set-{random_state}.csv"
        classy = data_state(random_state)
        y_train, y_pred_train, y_test, y_pred_test, y_pred_test_prob =classy.native_bayes_total()
        classy.area_under_roc_curve(y_test, y_pred_test_prob, "NB ROC Curve", save_png = True)


    print("\n\nDone")
