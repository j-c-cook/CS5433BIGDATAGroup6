# Group 6 Task 2 Code
# Friday, April 30, 2021

# Jack Cook
# Thomas Okonkwo
# Govardhan Digumurthi

"""
This file must be executed using `spark-submit`:

spark-submit Group6_Task_1_Code.py Group6_Task_1_Output.csv

The following is what is done in this file:
- The spark configuration is defined
- The `Group6_Task_1_Output.csv` is read in from the command line into
  a spark dataframe
- The spearman correlation is computed between all the features versus
  the independent variable; price (done using pyspark built in
  function)
- The dataframe is converted to a pandas dataframe to further
  investigate correlations, the Pearson, Spearman and Kendall methods
  are computed. The following is done for each method:
  * The correlation is computed
  * The correlation dataframe is exported to a csv
  * The correlation dataframe is printed to the console
  * A heat map plot is created and saved
"""

from pyspark import SparkConf, SparkContext
import sys
from pyspark.sql import SQLContext
import seaborn as sb
import matplotlib.pyplot as plt


def pb(sentance):
	# Provide a print break for console
	print(71 * '-')
	print(sentance)
	print(71 * '-')
	return


# ---------------------------------------------------------------------
pb('Spark Configuration setup')
# ---------------------------------------------------------------------

# Give the spark configuration
conf = SparkConf().setMaster("local").setAppName("Group6Task2Code")
# Set the log level to "WARN" to hide all INFO lines printed out
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlsc = SQLContext(sc)


# ---------------------------------------------------------------------
pb('Read csv file in from command line to dataframe')
# ---------------------------------------------------------------------

# get the file location from the command line arg
file_location = sys.argv[1]
file_type = file_location.split('.')[-1]  # get the file type
if file_type != 'csv':  # if its not a csv warn that it should be
    raise ValueError('Expected a csv file.')

infer_schema="true"
first_row_header="true"
delimiter=","

# Load the csv file using SQLContext
df = sqlsc.read.format(file_type)\
	.option("inferSchema",infer_schema)\
	.option("header",first_row_header)\
	.option("sep",delimiter)\
	.load(file_location)


# ---------------------------------------------------------------------
pb('Calculate the Pearson correlation between features and price')
# ---------------------------------------------------------------------

# This is done using the built in function corr and only does Pearson
# The output is price
independent_variable = 'Price'

# Get all column names
col_names = df.schema.names
# Remove the column that pandas ads in
col_names = [col_names[i] for i in range(len(col_names)) if 
	     col_names[i] != '_c0']

df = df.select(*col_names)

print('Independent\tDependent\tPearson')
for _, dependent_var in enumerate(col_names):
	_corr = df.stat.corr(independent_variable, 
			     dependent_var, 
    			     method='pearson')
	print('{0}\t{1}\t{2:.4f}'.\
	      format(independent_variable, dependent_var, _corr))

# ---------------------------------------------------------------------
pb('Compute correlations using pandas')
# ---------------------------------------------------------------------

# Pyspark only has the Pearson correlation built in, so we will look
# at other correlations using pandas

# convert the dataframe to pandas
df_pd = df.toPandas()

# Create independent dataframe
df_pd_y = df_pd[independent_variable]

# ---------------------------------------------------------------------
pb('Compute Pearson correlation in pandas and plot heat map')
# ---------------------------------------------------------------------

# Pearson Method
df_pearson = df_pd.corr(method='pearson')

# export the dataframe to a csv
df_pearson.to_csv('Group6_Task_2_Output_Pearson.csv')

# print to the console with 4 decimals
print(df_pearson.round(decimals=4))
# round to 2 decimals for heatmap
df_pearson = df_pearson.round(decimals=2)

sb.set(font_scale=0.8)

# Plot the heat map for pearson
sns_plot = sb.heatmap(df_pearson, 
           xticklabels=df_pearson.columns, 
           yticklabels=df_pearson.columns, 
           cmap='RdBu_r', 
           annot=True, 
           linewidth=4)

fig = sns_plot.get_figure()

plt.tight_layout()

fig.savefig('Group6_Task_2_Output_Pearson_heatmap.pdf')

plt.close(fig)

# ---------------------------------------------------------------------
pb('Compute Spearman correlation in pandas and plot heat map')
# ---------------------------------------------------------------------

# Calculate the correlation using the spearman method
df_spearman = df_pd.corr(method='spearman')

# Export the dataframe to a csv
df_spearman.to_csv('Group6_Task_2_Output_Spearman.csv')

# print consol with 4 decimal places
print(df_spearman.round(decimals=4))

# round to 2 decimal for heat map
df_spearman = df_spearman.round(decimals=2)

# Plot the heat map for spearman
sns_plot = sb.heatmap(df_spearman, 
           xticklabels=df_spearman.columns, 
           yticklabels=df_spearman.columns, 
           cmap='RdBu_r', 
           annot=True, 
           linewidth=4)

fig = sns_plot.get_figure()

plt.tight_layout()

fig.savefig('Group6_Task_2_Output_Spearman_heatmap.pdf')

plt.close(fig)


# ---------------------------------------------------------------------
pb('Compute Kendall correlation in pandas and plot heat map')
# ---------------------------------------------------------------------

# Calculate the correlation using the kendall method
df_kendall = df_pd.corr(method='kendall')

# Export the dataframe to a csv
df_kendall.to_csv('Group6_Task_2_Output_Kendall.csv')

# Print to console with 4 decimal places
print(df_kendall.round(decimals=4))

# round to 2 decimals for heat map
df_kendall = df_kendall.round(decimals=2)

# Plot the heat map for spearman
sns_plot = sb.heatmap(df_kendall, 
           xticklabels=df_kendall.columns, 
           yticklabels=df_kendall.columns, 
           cmap='RdBu_r', 
           annot=True, 
           linewidth=4)

fig = sns_plot.get_figure()

plt.tight_layout()

fig.savefig('Group6_Task_2_Output_Kendall_heatmap.pdf')

plt.close(fig)
