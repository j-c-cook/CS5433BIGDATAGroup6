import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#importing the housing data and removed the last 18 rows of the dataset and saving to a new csv file

pd.read_csv('Housing_data-Final-1.csv')[:-18].to_csv(r'housing-data.csv', index = False)

#reading the new csv file into a dataframe
df = pd.read_csv('housing-data.csv')

#showing the top 5 elements of the table
df.head()

#calculate the correlation using the pearson method
df_pearson = df.corr(method="pearson")
df_pearson 

#Lets plot a graph and see how this looks like on graph
sb.heatmap(df_pearson, 
            xticklabels=df_pearson.columns,
            yticklabels=df_pearson.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


#calculate the correlation using the spearman method
df_spearman = df.corr(method="spearman")
df_spearman


#Lets plot a graph and see how this looks like on graph
sb.heatmap(df_pearson, 
            xticklabels=df_spearman.columns,
            yticklabels=df_spearman.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


#calculate the correlation using the Kendall method
df_kendall = df.corr(method="kendall")
df_kendall


#Lets plot a graph and see how this looks like on graph

sb.heatmap(df_kendall, 
            xticklabels=df_kendall.columns,
            yticklabels=df_kendall.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

#we have plotted 3 different method of correllation.
#If the goal is to predict what features will affect the price of the house or apartment, we can see from the three
#correllation method that number of bedroom "#Bedroom" will greatly affect the pricing. Now lets calculate individually the
#correllation of price against all the features in the dataset

#using price as the the goal or prediction value, we add it to a seperate dataframe
price_df = df["Price"]

#calculate the correllation coefficient with the pearson method
df.corrwith(price_df,method="pearson",axis=0)

#calculate the correllation coefficient with the spearman method
df.corrwith(price_df,method="spearman",axis=0)

#calculate the correllation coefficient with the kendall method
df.corrwith(price_df,method="kendall",axis=0)

## we can see that across all the three methods for correllation, bedroom and bathroom are the the sole reason why price
#of a particular housing will be expensive.

#However we can see that, the pearson correllation coefficient says otherwise. It says that the bathroom is responsible for
#increase in pricing unlike spearman and kendall method




