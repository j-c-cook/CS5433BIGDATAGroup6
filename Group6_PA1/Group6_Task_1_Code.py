# Group 6 Task 1 Code
# Tuesday, April 13, 2021

# Jack Cook
# Thomas Okonkwo
# Govardhan Digumurthi


"""
This file must be executed using `spark-submit`:

bin/spark-submit Group6_Task_1_Code.py path/to/file.csv


The following is a list of the tasks performed in this file in order:

- A spark configuration is defined
- A csv file is read in from a command line argument
- The dataframe is analyzed
  * The number of rows and columns are found
  * The `Schema` for the dataframe is printed to the console, so that
    the types in each column can be understood
  * The columns of type string are analyzed to see if there is any
    foul data
- The string values in the dataset are replaced with numeric values
- The dataframe is split into two dataframes
  * "mal" the bad dataframe contains null and less than 0 values
  * "gut" the good dataframe contains values that are okay for cosine
    similarity

The following are the examples listed at the bottom in order:

- A simple cosine similarity using dense user defined vectors to
  understand the function
- Compute cosine similarity of the first row versus all other rows
  in the data frame
- Normalize the data then compute cosine similarity of the first row
  versus all other rows
"""


from pyspark import SparkConf, SparkContext
import sys
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StringType, FloatType, StructType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as f
from functools import reduce
from pyspark.sql.functions import udf
import numpy as np


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
conf = SparkConf().setMaster("local").setAppName("Group6Task1Code")
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
pb('Analyze the dataframe to understand and describe data set')
# ---------------------------------------------------------------------

# Get the number of rows in the dataframe
n_rows = df.count()
print('The number of rows in the dataset: {}'.format(n_rows))
# Get the number of columns in the dataframe
n_cols = len(df.columns)
print('The number of columns in the dataframe: {}'.format(n_cols))

# Provide a description of the dataset in the console (print)
df.printSchema()

# For the string columns, we can take the set of each column to find the
# unique values
str_cols = ['Suburb', 'Type', 'Region_name']

# export each column containing type string to a csv by the name of that
# column
for i in range(len(str_cols)):
	str_col = str_cols[i]
	_df = df.select(str_col)
	df2 = _df.distinct()
	df2.toPandas().to_csv('{}.csv'.format(str_cols[i]))
# The Property_count is of type integer, and appears to be the same at
# each suburb
# Find the the distincts of 'Suburb' and 'Property_count'
_df = df.select('Suburb', 'Property_count')
df2 = _df.distinct()
df2.toPandas().to_csv('Suburb-Property_count.csv')
# There are 312 rows in 'Suburb.csv' and 'Suburb-Property_count.csv'

# ---------------------------------------------------------------------
pb('Modify the columns containing strings to contain numbers')
# ---------------------------------------------------------------------

# Index string columns to give them a numeric value from 0->n
# https://stackoverflow.com/a/65849758/11637415
# the index of the string values in the columns
indexers = [StringIndexer(inputCol=c, outputCol='{0}_indexed'.\
			  format(c)) for c in str_cols]

# Create a new dataframe with the string columns indexed
pipeline = Pipeline(stages=indexers)
df2 = pipeline.fit(df).transform(df)
df2.toPandas().to_csv('strings_indexed.csv')

# Select all non-string columns and create a new df
cols = df2.schema.names
cols = [cols[i] for i in range(len(cols)) if cols[i] not in str_cols]
df3 = df2.select(*cols)
df3.toPandas().to_csv('strings_indexed_reduced.csv')

# At this point we need all of the columns to be of type float
from pyspark.sql.functions import col
for col_name in cols:
	df3 = df3.withColumn(col_name, col(col_name).cast('float'))

df3.printSchema()

# WORKING WITH DF3 NOW

# ---------------------------------------------------------------------
pb('Create a new seperated dataframe that contain out of range values')
# ---------------------------------------------------------------------
# The out of range values are 'null' and numeric values less than 0

# Find the null values
df_null = df3.where(reduce(lambda x, y: x | y, (f.col(x).isNull() \
                           for x in df3.columns)))
print('The number of rows containing null values: {}'.\
       format(df_null.count()))
# Find the values that are less than 0
df_leq_0 = df3.where(reduce(lambda x, y: x | y, (f.col(x) < 0 \
                            for x in df3.columns)))
print('The number of rows containing < 0: {}'.\
      format(df_leq_0.count()))
# Combine the null and less than 0 rows into one mal dataframe
df4_mal = df_null.unionAll(df_leq_0)
print('The mal-dataframe')
df4_mal.show()
n_rows_mal = df4_mal.count()
print('Total number of mal rows: {}'.format(n_rows_mal))

# Create a good ("gut") dataframe
# Drop rows in the dataframe that contain NULL values
df_not_null = df3.dropna()

# Remove all values that are less than 0
# https://stackoverflow.com/a/64530760/11637415
df4_gut = df_not_null.exceptAll(df_leq_0)

n_rows_gut = df4_gut.count()
print('Good dataframe count: {}'.format(n_rows_gut))

# Check to make sure we haven't lost any rows
total_rows = n_rows_mal + n_rows_gut
print('The total rows in the good and bad dataframes are: {}'.\
						    format(total_rows))

# ---------------------------------------------------------------------
pb('Define functions for normalizing a dataframe')
# ---------------------------------------------------------------------
def normalize_formula(df, averages, std_devs, column):
	# norm = [(X - mean) / std_dev]
	df = df.withColumn(column + '_norm', \
                          ((df[column] - averages[column]) \
                            / std_devs[column + '_stddev']))
	return df


def compute_normalization_statistics(df, columns):
	"""
	Input a pyspark dataframe, return the averages and standard 
	deviations for each column.
	"""
	# find the mean and standard deviation for each column in
	# train_df
	aggExpr = []
	aggStd = []
	for column in columns:
		aggExpr.append(f.mean(df[column]).alias(column))
		aggStd.append(f.stddev(df[column]).\
			      alias(column + '_stddev'))

	averages = df.agg(*aggExpr).collect()[0]
	std_devs = df.agg(*aggStd).collect()[0]
	return averages, std_devs

def normalize_for_each_column(df, columns, averages, std_devs):
	"""
	Loop overa all of the columns and create normalized columns
	based on the input list of columns.
	"""
	# normalize each dataframe, column by column
	for column in columns:
		df = normalize_formula(df, averages, std_devs, column)
	return df


def normalize(train_df, test_df, columns):
	"""
	Normalize the columns of the testing and training dataframe with
	mean and standard deviation computed using the training
	dataframe. This results in no data leakage.

	norm = [(X - mean) / std_dev ]

	Inspiration: Morgan McGuire
	URL: https://gist.github.com/morganmcg1/15a9de711b9c5e8e1bd142b4be80252d#file-pyspark_normalize-py
	"""
	# compute averages and standard deviation for training df
	averages, std_devs = compute_normalization_statistics(train_df, columns)

	# normalize each dataframe, column by column
	# normalize the training dataframe
	train_df = normalize_for_each_column(train_df, 
					     columns, 
				   	     averages, 
					     std_devs)
	# normalize the testing dataframe, using training df statistics
	test_df = normalize_for_each_column(test_df, 
					    columns, 
					    averages, 
					    std_devs)

	return train_df, test_df, averages, std_devs


# ---------------------------------------------------------------------
pb('Define useful re-usable functions for accessing dataframes')
# ---------------------------------------------------------------------

def indexing_function(df, col_name='id'):
	# Create an indexing column by name "id"
	# https://stackoverflow.com/a/37490920/11637415
	return df.withColumn(col_name, f.monotonically_increasing_id())


def vector_assemble_function(df, inputCols, outputCol='Vector'):
	"""
	Assemble the given columns into a column named 'vector'

	inputCols : list
		a list of the input columns
	outputCol : string
		a string defining the column where the vector will be
		stored
	"""
	# if outputCol already exists in dataframe, remove it
	if outputCol in df.columns:
		df = df.drop(outputCol)

	# Create vectors based on column list
	vecAssembler = VectorAssembler(inputCols=inputCols, 
			               outputCol=outputCol, 
			               handleInvalid='keep')
	df = vecAssembler.transform(df)
	return df


def select_cell_by_id(df, id_num, col_name='Vector'):
	"""
	col_name : string
		The name of the column to select from
	id_num : int
		The id by row to select
	"""
	# https://stackoverflow.com/a/64588611/11637415
	return df.filter(col('id') == id_num).select(col_name).head()[0]


def find_row_max(df, colName='coSim'):
	"""
	Return the row in which the max value in the given column name 
	occurs.
	"""
	maxVal = df.agg(f.max(colName)).collect()[0][0]
	row = df.filter(f.col(colName) == maxVal).first()
	return row, maxVal

# ---------------------------------------------------------------------
pb('Define functions for computing cosine similarity')
# ---------------------------------------------------------------------

# http://grahamflemingthomson.com/cosine-similarity-spark/
# write user defined function for cosine similarity
def cos_sim(a, b):
        return float(np.dot(a, b) / \
                              (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_cosine_similarity(df, single_vector, inputCol='Vector', 
			      outputCol='coSim'):
	"""
	Compute the cosine similarity of a single vector versus a 
	column containing vectors in a dataframe.
	"""
	# Remove the coSim column if it already exits
	if outputCol in df.columns:
		df = df.drop(outputCol)

	df = df.withColumn('coSim', udf(cos_sim, FloatType())\
			   (f.col('Vector'), f.array([f.lit(v) \
                                             for v in single_vector])))
	return df

# ---------------------------------------------------------------------
pb('Use cosine similarity to replace bad values in mal dataframes')
# ---------------------------------------------------------------------

# Give gut and mal dataframes an id column
df4_gut_mod = indexing_function(df4_gut, col_name='id')
df4_mal = indexing_function(df4_mal, col_name='id')

df4_mal.show()

# get column names
col_names = df4_gut.schema.names

# get all columns but the "id"
col_names = [col_names[i] for i in range(len(col_names)) \
	     if col_names[i] != 'id']
print(col_names)

# Get normalization statistics from gut dataframe
averages, std_devs = compute_normalization_statistics(df4_gut_mod, 
						      col_names)

def find_similar(row, col_names):
	bad_col = None
	bad_val = None
	idx = row['id']
	for j in range(len(col_names)):
		a = row[col_names[j]]
		a_type = type(a)
		if a_type is float:
			if a < 0:
				bad_col = col_names[j]
				bad_val = a
		elif a_type is type(None):
			bad_col = col_names[j]
			bad_val = a
	return bad_col, bad_val, idx

bad_columns = []
indices = []
mal_vals = []

for row in df4_mal.rdd.collect():
	bad_col, mal_val, idx = find_similar(row, col_names)
	if type(bad_col) is type(None):
		pass
	else:
		bad_columns.append(bad_col)
		indices.append(idx)
		mal_vals.append(mal_val)

print(list(zip(indices, bad_columns)))

def find_cosine_similarity_replacement(df_gut, df_mal, mal_col, idx, mal_val, averages, std_devs):
	
	# Assemble columns to be used in normalization
	columns_for_norm = [col_names[i] for i in range(len(col_names)) if col_names[i] != bad_col]
	
	# Apply normalizatin statistics and compute normalization of df
	df_gut = normalize_for_each_column(df_gut, columns_for_norm, 
                                    averages, std_devs)
	# Select the columns for normalization and the "id" in the mal df
	#df_mal = df_mal.select(*columns_for_norm, 'id')
	# Apply normalization statistics to mal column
	df_mal = normalize_for_each_column(df_mal, columns_for_norm,
					     averages, std_devs)

	# Assemble a column of vectors from the '_norm' columns
	col_names_vectors = [columns_for_norm[i] + '_norm' \
                     for i in range(len(columns_for_norm))]
	# Create a column of vectors that don't include the bad column
	df_gut = vector_assemble_function(df_gut, col_names_vectors)
	# Assemble vector for mal df
	df_mal = vector_assemble_function(df_mal, col_names_vectors)
	# Select the vector in the indexed row from the mal df
	f1 = select_cell_by_id(df_mal, idx, col_name='Vector')
	# Compute the cosine similarity between the vector and all rows
	df_gut = compute_cosine_similarity(df_gut, f1)
	
	# Find index where the cosine similarity is maximum
	row, max_coSim = find_row_max(df_gut, colName='coSim')
	row_idx = row['id']
	# print('row: {}\tmax_coSim: {}'.format(row, max_coSim))
	# df4_gut_mod.toPandas().to_csv('df4_gut_tmp.csv')
	new_value = select_cell_by_id(df_gut, row_idx, col_name=bad_col)	
	return new_value, row_idx
	



bad_col = bad_columns[0]
idx = indices[0]
mal_val = mal_vals[0]
print(mal_vals)

new_value, row_idx = find_cosine_similarity_replacement(df4_gut_mod, df4_mal, bad_col, idx, mal_val, averages, std_devs)

print('({}, {})\t{}\t({}, {})'.format(idx, bad_col, mal_val, row_idx, new_value))

t = (f.when(f.col('id') == idx, new_value))

df4_fixed = df4_mal.filter(df4_mal['id'] == idx).withColumn(bad_col, t)

for i in range(1, len(bad_columns)):
	bad_col = bad_columns[i]
	idx = indices[i]
	mal_val = mal_vals[i]

	new_value, row_idx = find_cosine_similarity_replacement(df4_gut_mod, df4_mal, bad_col, idx, mal_val, averages, std_devs)

	print('({}, {})\t{}\t({}, {})'.format(idx, bad_col, mal_val, row_idx, new_value))

	t = (f.when(f.col('id') == idx, new_value))
	df4_fixed_i = df4_mal.filter(df4_mal['id'] == idx).withColumn(bad_col, t)
	df4_fixed = df4_fixed.union(df4_fixed_i)

df4_fixed.show()

# ---------------------------------------------------------------------
pb('Combine good and previously bad dataframe for export')
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
pb('--------------------------- Examples ----------------------------')
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
pb('Simple cosine similarity')
# ---------------------------------------------------------------------

def cosine_similarity(v1, v2):
	return v1.dot(v2) / (v1.norm(2) * v2.norm(2))

# https://stackoverflow.com/a/46751471/11637415
# Example of low cosine similarity
x = Vectors.dense([1, 2, 3])
y = Vectors.dense([9, 8, 4])

a = cosine_similarity(x, y)

print('Cosine Similarity of x and y: {}'.format(a))

# Example of high cosine similarity

b = cosine_similarity(x, x)

print('Cosine Similarity of x and x: {}'.format(b))

# ---------------------------------------------------------------------
pb('Cosine similarity without normalization')
# ---------------------------------------------------------------------

# Get column names from good dataframe
example_cols = df4_gut.schema.names

df_example = indexing_function(df4_gut, col_name='id')

df_example = vector_assemble_function(df_example, 
				      example_cols, 
				      outputCol='Vector')

f1 = select_cell_by_id(df_example, 0, col_name='Vector')
print('The first row in the data frame to be compared using Cosine '
      'Similarity to all other rows in the data frame.')
print(f1)

df_example = compute_cosine_similarity(df_example, f1)

df_coSim = df_example.select(f.col('coSim'))
df_coSim.summary('count', 'min', 'stddev', 'max').show()

row_1, max_coSim = find_row_max(df_example, colName='coSim')

print('A max cosine similarity of {} was found in row {}'.\
						   format(max_coSim, 
						          row_1['id']))

# ---------------------------------------------------------------------
pb('Cosine similarity with normalization')
# ---------------------------------------------------------------------

df = indexing_function(df4_gut)

# get column names
col_names = df.schema.names

# get all columns but the "id"
col_names = [col_names[i] for i in range(len(col_names)) \
	     if col_names[i] != 'id']

# Get normalization statistics
averages, std_devs = compute_normalization_statistics(df, 
						      col_names)
# Apply normalizatin statistics and compute normalization of df
df = normalize_for_each_column(df, col_names, averages, std_devs)

# Assemble a column of vectors from the '_norm' columns
col_names_vectors = [col_names[i] + '_norm' \
                     for i in range(len(col_names))]
df = vector_assemble_function(df, col_names_vectors)

# pull out vector from row 0
f1 = select_cell_by_id(df, 0, col_name='Vector')

# compute the cosine similarity between the vector and all rows
df = compute_cosine_similarity(df, f1)

# select only the coSim column and display statistics
df_coSim = df.select(f.col('coSim'))
df_coSim.summary('count', 'min', 'stddev', 'max').show()

row, max_coSim = find_row_max(df, colName='coSim')

print('A max cosine similarity of {} was found in row {}'.\
                                                   format(max_coSim, 
                                                          row_1['id']))

# ---------------------------------------------------------------------
pb('Statistics of dataframe before and after normalization')
# ---------------------------------------------------------------------
print('Look at statistics for good dataframe before and after' 
      ' normalization')
output_file_name = 'df_gut_summary.csv'
print('Dataframe (gut) summary exported to {}'.\
			format(output_file_name))
_df = df.select(*col_names)
_df.summary('count', 'min', 'stddev', 'max', 'mean').toPandas().\
			                     to_csv(output_file_name)
output_file_name = 'df_gut_normalized_summary.csv'
print('Dataframe (gut normalized) summary exported to {}'.\
                        format(output_file_name))
_df = df.select(*col_names_vectors)
_df.summary('count', 'min', 'stddev', 'max', 'mean').toPandas().\
				 	     to_csv(output_file_name)
