import torch
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt 
import seaborn as sns


class StandardScaler():

    def __init__(self, data):

        self.data = data
        self.mean = data.mean(0, keepdim = True)
        self.std = data.std(0, unbiased = False, keepdim = True)
    
    def forward(self):

        self.data -= self.mean
        self.data /= self.std
        return self.data




class rank_bar_line():

    def __init__(self, data, title = 'Plot', xaxis = "X-axis", yaxis = "y-axis", color_palette = 'summer'):
        
        self.data = data
        self.title = title
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.color_palette = color_palette

        self.plot_chart()
    
    def plot_chart(self):

        plt.title(self.title)
        sns.barplot(x=self.xaxis, y=self.yaxis, data=self.data, palette=self.color_palette)
        plt.show()