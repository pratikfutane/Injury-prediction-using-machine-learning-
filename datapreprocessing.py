# python module imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class DataPreProcessing:
    def __init__(self):
        self.inj_df = pd.read_csv(os.getcwd()+"\\datasets\\injuries.csv")
        self.gmw_df = pd.read_csv(os.getcwd()+"\\datasets\\game_workload.csv")
        self.met_df = pd.read_csv(os.getcwd()+"\\datasets\\metrics.csv")
        self.mgd_df = pd.read_csv(os.getcwd() + "\\datasets\\mergedData.csv")
    def getDatasets(self):
        return self.inj_df, self.gmw_df, self.met_df, self.mgd_df
    def dataConversion(self):
        self.met_df['date'] = self.met_df['date'].astype('datetime64[ns]')
        self.gmw_df['date'] = self.gmw_df['date'].astype('datetime64[ns]')
        self.inj_df['date'] = self.inj_df['date'].astype('datetime64[ns]')
        self.inj_df["injury"] = "Yes"
        games_data = pd.merge(self.gmw_df, self.inj_df, how='left', left_on=['athlete_id', 'date'],
                              right_on=['athlete_id', 'date'])
        merged_data = games_data
        games_data["injury"].fillna("No", inplace=True) # return (head,shape)
        new_metrics_df = self.met_df.pivot_table('value', ['athlete_id', 'date'], 'metric').\
            reset_index() # return (head,shape)
        final_data = pd.merge(games_data, new_metrics_df, how='left', left_on=['athlete_id', 'date'],
                              right_on=['athlete_id', 'date']) # return (head,shape)
        final_data['rest_period'] = final_data.groupby('athlete_id')['date'].diff()
        final_data['rest_period'] = final_data['rest_period'].astype('timedelta64[D]') # return (head,Shape)
        final_data.corr()
        final_data.injury.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
        final_data = final_data[
            ['injury', 'athlete_id', 'date', 'game_workload', 'groin_squeeze', 'hip_mobility', 'rest_period']]
        # return head
        columns = ['injury','game_workload', 'groin_squeeze', 'hip_mobility', 'rest_period']
        dummy_variables = pd.get_dummies(final_data['athlete_id'])
        ready_data = pd.concat([final_data, dummy_variables], axis=1)
        ready_data.drop('athlete_id', axis=1, inplace=True)
        ready_data.drop('date', axis=1, inplace=True) # head
        print(ready_data.shape)
        # print(ready_data.head(10))
        # ready_data.apply(lambda x: sum(x.isnull()), axis=0)
        #print(ready_data.isnull().values.any())
        for c in columns:
            print(c,ready_data[c].isnull().sum())
        ready_data['rest_period'].fillna(ready_data['rest_period'].mean(), \
                                inplace=True)
        for c in columns:
            print(c,ready_data[c].isnull().sum())
        ready_data1 = ready_data
        # Assigning X and y values
        """Assiging the dependent variable (0 0r 1 to be predicted as class)"""
        X = ready_data.iloc[:, 1:35].values
        y = ready_data.loc[:, 'injury'].values
        y = y.reshape(-1, 1)
        # from collections import Counter
        # from sklearn.datasets import make_classification
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=0)
        return merged_data, games_data, final_data, ready_data1

    def classificationModels(self):
        # Logistic Regression
        models = ['LR', 'DT', 'KNN', 'SVM']
        accuracies = []
        colors = ['lightcoral','lightseagreen','orange','forestgreen']

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracies.append(model.score(self.X_test, self.y_test) * 100)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            model.score(self.X_test, self.y_test)))
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(self.y_test, y_pred)
        print(confusion_matrix)
        from sklearn.metrics import classification_report
        print(classification_report(self.y_test, y_pred))

        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracies.append(model.score(self.X_test, self.y_test) * 100)
        print('Accuracy of Decision Tree classifier on test set: {:.2f}'.\
              format(model.score(self.X_test, self.y_test)))
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(self.y_test, y_pred)
        print(confusion_matrix)
        print(classification_report(self.y_test, y_pred))

        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracies.append(model.score(self.X_test, self.y_test) * 100)
        print('Accuracy of Knn classifier on test set: {:.2f}'.format(model.score(self.X_test, self.y_test)))
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(self.y_test, y_pred)
        print(confusion_matrix)
        from sklearn.metrics import classification_report
        print(classification_report(self.y_test, y_pred))

        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracies.append(model.score(self.X_test, self.y_test) * 100)
        print('Accuracy of Support Vector classifier on test set: {:.2f}'.\
              format(model.score(self.X_test, self.y_test)))
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(self.y_test, y_pred)
        print(confusion_matrix)
        from sklearn.metrics import classification_report
        print(classification_report(self.y_test, y_pred))

        # plt.grid(True)
        #
        # # for index, value in enumerate(accuracies):
        # #     plt.text(value, index, str(value))
        # plt.bar(models, accuracies, color=colors)
        # plt.title("Comparitive Analysis of Classification Methods in Predicting Injuries")
        # plt.ylabel('Model')
        # plt.xlabel('Accuracy Percentage')
        # plt.savefig(os.getcwd() + "\\static\\graphs\\comp.png")
        # plt.show()
        return models, accuracies
dp = DataPreProcessing()
dp.dataConversion()
m, a = dp.classificationModels()
print(m)
print(a)

