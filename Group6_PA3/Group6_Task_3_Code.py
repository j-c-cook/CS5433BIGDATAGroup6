# Group 6 Task 3 Code
# Tuesday, April 24, 2021

# Jack Cook
# Thomas Okonkwo
# Govardhan Digumurthi


"""
This file must be executed using `spark-submit`:

bin/spark-submit Group6_Task_1_Code.py path/to/file.csv

<<<Provide an overview of the tasks performed in this script>>>

"""


import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType


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

col_names = df.schema.names
# remove the column that pandas placed in
col_names = [col_names[i] for i in range(len(col_names)) \
             if col_names[i] != '_c0']

df = df.select(*col_names)


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
pb('Split the data into training and test sets (80% train)')
# ---------------------------------------------------------------------
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/random_forest_regressor_example.py

(train_df, test_df) = df.randomSplit([0.8, 0.2])

# ---------------------------------------------------------------------
pb('Transform the dataframe to an RDD of LabeledPoint')
# ---------------------------------------------------------------------
# https://github.com/BhaskarBiswas/PySpark-Codes/blob/master/Random_Forest_pyspark.py

# Convert the dataframes to relational distributed databases
# train_rdd = train_df.rdd
# test_rdd = test_df.rdd

# Transform the rdd's into df's of LabeledPoint
# transformed_train_df = train_rdd.map(lambda row: LabeledPoint(row[0], Vectors.dense(row[1:])))
# transformed_test_df = test_rdd.map(lambda row: LabeledPoint(row[0], Vectors.dense(row[1:])))

col_names_vector = [col_names[i] for i in range(len(col_names)) \
                    if col_names[i] != 'Price']

# train_df, test_df, averages, std_devs = normalize(train_df, test_df, col_names_vector)

# col_names_vector = [col_names_vector[i] + '_norm' for i in range(len(col_names_vector))]

col_names_vector = ['#Bathroom']

train_df = vector_assemble_function(train_df, col_names_vector)
test_df = vector_assemble_function(test_df, col_names_vector)


# ---------------------------------------------------------------------
pb('Training a random Forest model on the dataset')
# ---------------------------------------------------------------------


rf = RandomForestRegressor(featuresCol='Vector', labelCol='Price', \
			   predictionCol='Prediction', maxDepth=30, numTrees=5000)

model = rf.fit(train_df)

predictions = model.transform(train_df)

predictions.select("Prediction", "Price", "Vector").show(5)


# Compute rmse
# https://stackoverflow.com/a/61176108/11637415

df = predictions.withColumn('difference', f.col('Price') - f.col('Prediction'))
df = df.withColumn('squared_difference', f.pow(f.col('difference'), f.lit(2).astype(IntegerType())))
rmse = df.select(f.sqrt(f.avg(f.col('squared_difference'))).alias('rmse'))

rmse.show()




