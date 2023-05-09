# python module imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

class SpInMLViz:
    def __init__(self):
        self.inj_df = pd.read_csv(os.getcwd()+"\\datasets\\injuries.csv")
        self.gmw_df = pd.read_csv(os.getcwd()+"\\datasets\\game_workload.csv")
        self.met_df = pd.read_csv(os.getcwd()+"\\datasets\\metrics.csv")
        self.mgd_df = pd.read_csv(os.getcwd() + "\\datasets\\mergedData.csv")
    def getDatasets(self):
        return self.inj_df, self.gmw_df, self.met_df, self.mgd_df

    def viz1(self):
        self.inj_df['date'] = self.inj_df['date'].astype('datetime64[ns]')
        injury_counts = self.inj_df.groupby("athlete_id").count().reset_index()
        injury_counts = injury_counts.rename(columns={'date': 'total_injuries'})
        rcParams['figure.figsize'] = 10, 4
        # plt.bar(injury_counts["athlete_id"], injury_counts["total_injuries"])
        #
        # plt.xticks(injury_counts["athlete_id"])
        # plt.legend(['total_injuries'])
        # plt.grid(True)
        # plt.title("Total Number of Injures occured by players")
        # plt.xlabel('Athlete ids')
        # plt.ylabel('Number of Injuries')
        # plt.savefig(os.getcwd()+"\\static\\graphs\\g1.png")
        #plt.show()
        # self.inj_df["injury"] = "Yes"
        # self.gmw_df['date'] = self.gmw_df['date'].astype('datetime64[ns]')
        # rcParams['figure.figsize'] = 10, 8
        # self.gmw_df.groupby("athlete_id")['game_workload'].mean().sort_values().\
        #     plot(kind='barh', y='workload',title="Average Workload by players",color='g')
        # plt.grid(True)
        # plt.savefig(os.getcwd() + "\\static\\graphs\\g2.png")
        # plt.show()

        # rcParams['figure.figsize'] = 5, 8
        # self.gmw_df.groupby("athlete_id")['date'].count().sort_values().plot(kind='barh', y='date', x="athlete_id",
        #                                                                      title="Number of Games played by each Athlete",
        #                                                                      color='r')
        # plt.grid(True)
        # plt.savefig(os.getcwd() + "\\static\\graphs\\g3.png")
        # plt.show()
    def mergedData(self):
        self.inj_df["injury"] = "Yes"
        games_data = pd.merge(self.gmw_df, self.inj_df, how='left', \
                              left_on=['athlete_id', 'date'],right_on=['athlete_id', 'date'])
        games_data['injury'].fillna("No", inplace=True)

        print(games_data.head())
        games_data['date'] = games_data['date'].astype('datetime64[ns]')
        games_data.dtypes
        games_data[(games_data.athlete_id == 13) & (games_data.injury == "Yes")]['game_workload']
        rcParams['figure.figsize'] = 10, 4
        for i in range(1, 31):
            plt.plot(games_data[games_data.athlete_id == i]['date'],
                     games_data[games_data.athlete_id == i]['game_workload'],
                     color='blue', linewidth=0.7)
            plt.plot(games_data[(games_data.athlete_id == i) & (games_data.injury == "Yes")]['date'],
                     games_data[(games_data.athlete_id == i) & (games_data.injury == "Yes")]['game_workload'],
                     color='red', linewidth=1, marker='o')
            plt.grid()
            plt.legend(['game_workload', 'injury'])
            plt.xlabel('Timescale')
            plt.ylabel('Amount of Workload')
            plt.title('Workload and Injury over time for Athlete id:' + str(i))
            plt.grid(True)
            plt.savefig(os.getcwd() + "\\static\\graphs\\g4.png")
            plt.show()
    def metricsViz(self):
        metrics_df = self.met_df.pivot_table('value', ['athlete_id', 'date'], 'metric').reset_index()
        metrics_df.head()
        #metrics_df['date'] = metrics_df['date'].astype('datetime64[ns]')
        metrics_df.dtypes

        self.inj_df["injury"] = "Yes"
        games_data = pd.merge(self.gmw_df, self.inj_df, how='left', \
                              left_on=['athlete_id', 'date'], right_on=['athlete_id', 'date'])
        games_data['injury'].fillna("No", inplace=True)
        final_data = pd.merge(games_data, metrics_df, how='left', left_on=['athlete_id', 'date'],
                              right_on=['athlete_id', 'date'])
        final_data.head()
        rcParams['figure.figsize'] = 10, 4
        for i in range(1, 31):
            plt.plot(final_data[final_data.athlete_id == i]['date'],
                     final_data[final_data.athlete_id == i]['groin_squeeze'],
                     color='green', linewidth=0.7)
            plt.plot(final_data[(final_data.athlete_id == i) & (final_data.injury == "Yes")]['date'],
                     final_data[(final_data.athlete_id == i) & (final_data.injury == "Yes")]['groin_squeeze'],
                     color='red', linewidth=1, marker='o')
            plt.grid()
            plt.legend(['groin_squeeze', 'injury'])
            plt.xlabel('Timescale')
            plt.ylabel('Amount of groin_squeeze')
            plt.title('Groin_squeeze and Injury over time for Athlete id:' + str(i))
            plt.grid(True)
            plt.savefig(os.getcwd() + "\\static\\graphs\\g5.png")
            plt.show()
    def viz3(self):
        rcParams['figure.figsize'] = 10, 4
        self.gmw_df['date'] = self.gmw_df['date'].astype('datetime64[ns]')
        self.gmw_df['year'] = self.gmw_df['date'].map(lambda x: x.strftime('%Y'))
        self.gmw_df['month'] = self.gmw_df['date'].map(lambda x: x.strftime('%m'))
        athletes = self.gmw_df['athlete_id'].unique()
        for athlete in range(1, 31):
            fig = plt.figure()
            self.gmw_df[self.gmw_df.athlete_id == athlete].groupby(['year', 'month', ])['athlete_id'].count().plot(
                kind="bar")
            plt.title('Number of games played per month by Athelete Id ' + str(athlete))
            plt.yticks(np.arange(0, 11, 1))
            plt.xlabel('Months and Years')
            plt.ylabel('Number of matches')
            plt.tight_layout()  # tip(!)
        plt.grid(True)
        plt.savefig(os.getcwd() + "\\static\\graphs\\g6.png")
        plt.show()
# obj = SpInMLViz()
# obj.metricsViz()