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


# ---------------------------------------------------------------------
# Spark Configuration setup
# ---------------------------------------------------------------------

# Give the spark configuration
conf = SparkConf().setMaster("local").setAppName("Group6Task1Code")
# Set the log level to "WARN" to hide all INFO lines printed out
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlsc = SQLContext(sc)

# ---------------------------------------------------------------------
# Read csv file in from command line to dataframe
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
# Analyze the dataframe to understand and describe data set
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
# Modify the columns containing strings to contain numbers
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
# Create a new seperated dataframe that contain out of range values
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
df4_mal = df_null.union(df_leq_0)
print('The mal-dataframe')
df4_mal.show()
n_rows_mal = df4_mal.count()
print('Total number of mal rows: {}'.format(n_rows_mal))

# Create a good ("gut") dataframe
# Drop rows in the dataframe that contain NULL values
df_not_null = df3.dropna()

print('Not null count: {}'.format(df_not_null.count()))

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
# Normalize the dataframe that contains good values
# ---------------------------------------------------------------------
def normalize_formula(df, averages, std_devs, column):
	# norm = [(X - mean) / std_dev]
	df = df.withColumn(column, \
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
# Use cosine similarity to replace bad values in mal dataframes
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
	df = df.withColumn('coSim', udf(cos_sim, FloatType())\
			   (f.col('Vector'), f.array([f.lit(v) \
                                             for v in single_vector])))
	return df



# ---------------------------------------------------------------------
# --------------------------- Examples --------------------------------
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Simple cosine similarity
# ---------------------------------------------------------------------

def cosine_similarity(v1, v2):
	return v1.dot(v2) / (v1.norm(2) * v2.norm(2))

# https://stackoverflow.com/a/46751471/11637415
# Example of low cosine similarity
x = Vectors.dense([1, 2, 3])
y = Vectors.dense([9, 8, 4])

a = cosine_similarity(x, y)

print('cosine similarity: {}'.format(a))

# Example of high cosine similarity

b = cosine_similarity(x, x)

print('cosine similarity: {}'.format(b))

# ---------------------------------------------------------------------
# Cosine similarity without normalization
# ---------------------------------------------------------------------

# Get column names from good dataframe
example_cols = df4_gut.schema.names

def indexing_function(df, col_name='id'):
	# Create an indexing column by name "id"
	# https://stackoverflow.com/a/37490920/11637415
	return df.withColumn(col_name, f.monotonically_increasing_id())

df_example = indexing_function(df4_gut, col_name='id')

def vector_assemble_function(df, inputCols, outputCol='Vector'):
	"""
	Assemble the given columns into a column named 'vector'

	inputCols : list
		a list of the input columns
	outputCol : string
		a string defining the column where the vector will be
		stored
	"""
	# Create vectors based on column list
	vecAssembler = VectorAssembler(inputCols=inputCols, 
			               outputCol=outputCol, 
			               handleInvalid='keep')
	df = vecAssembler.transform(df)
	return df

df_example = vector_assemble_function(df_example, 
				      example_cols, 
				      outputCol='Vector')

def select_cell_by_id(df, id_num, col_name='Vector'):
	"""
	col_name : string
		The name of the column to select from
	id_num : int
		The id by row to select
	"""
	# https://stackoverflow.com/a/64588611/11637415
	return df.filter(col('id') == id_num).select(col_name).head()[0]
f1 = select_cell_by_id(df_example, 0, col_name='Vector')
print(f1)

df_example = compute_cosine_similarity(df_example, f1)
df_example.toPandas().to_csv('coSim_no_norm.csv')

# ---------------------------------------------------------------------
# Cosine similarity with normalization
# ---------------------------------------------------------------------

# TODO: normalize a column in a dataframe




# Cosine Similarity
# https://stackoverflow.com/a/46764347/11637415
from pyspark.ml.feature import HashingTF, IDF
# hashingTF = HashingTF(inputCol='Distance', outputCol='Distance_tf')
# tf = hashingTF.transform(df3)

# idf = IDF(inputCol='Distance_tf', outputCol='feature').fit(tf)
# ftidf = idf.transorm(tf)

# from pyspark.ml.feature import Normalizer
# normalizer = Normalizer(inputCol='Distance', outputCol='Distance_norm')
# data = normalizer.transform(df3)

# http://grahamflemingthomson.com/cosine-similarity-spark/
# write user defined function for cosine similarity
import numpy as np
def cos_sim(a, b):
	return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# from pyspark.ml.feature import VectorAssembler
# vecAssembler = VectorAssembler(inputCols='Price', outputCol='Price_vector')
# static_array = vecAssembler.transform(df3)

# df4 = df3.withColumn("coSim", udf(cos_sim, FloatType())(col('Price'), array([lit(v) for v in static_array])))

from pyspark.ml.linalg import Vectors

# https://stackoverflow.com/a/49832957/11637415


from pyspark.ml.feature import VectorAssembler

# pipeline = Pipeline(stages=vecAssem)
# df4 = pipeline.fit(df3).transform(df3)
# vecAssem.toPandas().to_csv('file.csv')

# https://stackoverflow.com/a/46764347/11637415
# Compute L2 Normalization of each column
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler

cols = df3.schema.names
# Iterating over columns to be scaled
# for i in cols:
    # VectorAssembler Transformation - Converting column to vector type
    # assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    # scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    # pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    # df3 = pipeline.fit(df3).transform(df3).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

# df3.toPandas().to_csv('df3.csv')



# from pyspark.ml.feature import VectorSlicer

# slicer = VectorSlicer(inputCol='Vector', outputCol='OneVector', indices=[0])

# output = slicer.transform(df4)

# output.select('Vector', 'OneVector').show()

# rdd = df6.rdd

# cc = rdd.take(1)

# print(cc)

# df7 = df4.withColumn('coSim', udf(cosine_similarity, FloatType())(col('Vector'), df6))


# Apply UDF to rows "when"
# https://stackoverflow.com/a/65823089/11637415
