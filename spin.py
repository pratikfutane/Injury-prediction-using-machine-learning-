# python module imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

class SpInML:
    def __init__(self):
        self.inj_df = pd.read_csv(os.getcwd()+"\\datasets\\injuries.csv")
        self.gmw_df = pd.read_csv(os.getcwd()+"\\datasets\\game_workload.csv")
        self.met_df = pd.read_csv(os.getcwd()+"\\datasets\\metrics.csv")
        self.mgd_df = pd.read_csv(os.getcwd() + "\\datasets\\mergedData.csv")
    def getDatasets(self):
        return self.inj_df, self.gmw_df, self.met_df, self.mgd_df
    def getConvertedData(self):
        self.inj_df['date'] = self.inj_df['date'].astype('datetime64[ns]')
        injury_counts = self.inj_df.groupby("athlete_id").count().reset_index()
        injury_counts = injury_counts.rename(columns={'date':'total_injuries'})
        injury_counts.sort_values("total_injuries", ascending = False,inplace= True)
        return injury_counts,self.gmw_df, self.met_df,self.mgd_df
