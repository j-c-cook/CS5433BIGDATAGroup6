# Group 6 Task 3 Code
# Tuesday, April 24, 2021

# Jack Cook
# Thomas Okonkwo
# Govardhan Digumurthi


"""
This file must be executed using `spark-submit`:

bin/spark-submit Group6_Task_1_Code.py path/to/Group6_Task_1_Output.csv

The following is a list in order of what happens in this program:
- The spark configuration is setup
- A file path to `Group6_Task_1_Output.csv` is read in from the command
  line
- Useful re-usable functions for accessing dataframes are defined
- The dataframe is randomly split into 80/20 train/test sets
- Columns of vectors are created that will contain features for the
  networks:
  * a) all features in the dataset
  * b) Only the feature with the highest correlatin as determined in
       task 2 (specifically #Bedroom)
- Two random forest regression models are created using the training
  data
- The two models and the datasets (training and testing) are exported
"""


import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, \
RandomForestRegressionModel
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

(train_df, test_df) = df.randomSplit([0.8, 0.2], 17)


# ---------------------------------------------------------------------
pb('Create columns containing vectors for parts a) and b)')
# ---------------------------------------------------------------------
# https://github.com/BhaskarBiswas/PySpark-Codes/blob/master/Random_Forest_pyspark.py

# All features in the dataset
col_names_vector_a = [col_names[i] for i in range(len(col_names)) \
                    if col_names[i] != 'Price']

# Only the feature with the highest correlation as determined in task 3
# above
col_names_vector_b = ['#Bathroom']

# Testing and training dataframes for A and B vectors
train_df = vector_assemble_function(train_df, col_names_vector_a, 
				    outputCol='VectorA')
test_df = vector_assemble_function(test_df, col_names_vector_a, 
				   outputCol='VectorA')

train_df = vector_assemble_function(train_df, col_names_vector_b, 
				    outputCol='VectorB')
test_df = vector_assemble_function(test_df, col_names_vector_b, 
				   outputCol='VectorB')


# ---------------------------------------------------------------------
pb('Train a random Forest model on the datasets')
# ---------------------------------------------------------------------

# Random forest models for parts a) and b)
rf_A = RandomForestRegressor(featuresCol='VectorA', labelCol='Price', 
			     predictionCol='Prediction', maxDepth=5, 
			     numTrees=500)

rf_B = RandomForestRegressor(featuresCol='VectorB', labelCol='Price', 
                             predictionCol='Prediction', maxDepth=5, 
                             numTrees=500)
# Fit the models
model_A = rf_A.fit(train_df)
model_B = rf_B.fit(train_df)


# ---------------------------------------------------------------------
pb('Export the RF models and the testing and training datasets')
# ---------------------------------------------------------------------

# File names for Forest Regression output
output_A = 'Group6_Task_3_Output_RF_A'
output_B = 'Group6_Task_3_Output_RF_B'

# Make sure there is not existing models
# If models exist, then remove them

import os

model_A_exists = os.path.isdir(output_A)
model_B_exists = os.path.isdir(output_B)

import shutil

if model_A_exists:
	shutil.rmtree(output_A)
if model_B_exists:
	shutil.rmtree(output_B)

# Export the models to a file
model_A.save(output_A)
model_B.save(output_B)

# Export testing and training dataframes
output_train = 'Group6_Task_3_Output_Train.csv'
output_test = 'Group6_Task_3_Output_Test.csv'

train_df.toPandas().to_csv(output_train)
test_df.toPandas().to_csv(output_test)

# -------------------------------------------
# EOF - the following will be moved to task 4

model = RandomForestRegressionModel.load(output_A)

predictions = model.transform(train_df)

predictions.select("Prediction", "Price", "VectorA").show(5)

# Compute rmse
# https://stackoverflow.com/a/61176108/11637415

df = predictions.withColumn('difference', f.col('Price') - f.col('Prediction'))
df = df.withColumn('squared_difference', f.pow(f.col('difference'), f.lit(2).astype(IntegerType())))
rmse = df.select(f.sqrt(f.avg(f.col('squared_difference'))).alias('rmse'))

rmse.show()




