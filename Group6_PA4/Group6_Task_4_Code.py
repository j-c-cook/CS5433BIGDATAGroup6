# Group 6 Task 4 Code
# Tuesday, April 27, 2021

# Jack Cook
# Thomas Okonkwo
# Govardhan Digumurthi

"""
This file must be executed using `spark-submit`:

bin/spark-submit Group6_Task_4_Code.py Group6_Task_3_Output_RF_A Group6_Task_3_Output_RF_B Group6_Task_3_Output_Test.csv

The following is a list of tasks done in this program:

- The spark configuration is setup
- Three file paths output from task 3 are read in from the command line
  * Path to random forest regression model a - read in as a random
    forest regression model
  * Path to random forest regression model b - read in as a random
    forest regression model
  * Testing dataset - read in as a pyspark sql dataframe
- Useful functions for accessing dataframes are defined
- Columns containing vectors for parts a and b are created (Note:
  this step is repeated in task 4 because storing vectors in an output
  file is no easy task, so the vectors are recreated here):
  * a) all features in the dataset
  * b) Only the feature with the highest correlation as determined in
       task 2 (specifically Bedroom)
- Both random forest regression models are used to make predictions
  on price
- Some of the predictions are shown from each dataframe
- The root mean squared error is computed for the all features and the
  bathroom model prediction, the rmse is shown in the console
"""

import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler
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
pb('Read files in from command line')
# ---------------------------------------------------------------------

# path to random forest model from task 3 part a)
rf_a_path = sys.argv[1]
# path to random forest model from task 3 part b)
rf_b_path = sys.argv[2]
# path to testing data set
test_df_path = sys.argv[3]

# read in the random forest regression models
rf_model_a = RandomForestRegressionModel.load(rf_a_path)
rf_model_b = RandomForestRegressionModel.load(rf_b_path)


file_type = test_df_path.split('.')[-1]  # get the file type
if file_type != 'csv':  # if its not a csv warn that it should be
    raise ValueError('Expected a csv file.')

infer_schema="true"
first_row_header="true"
delimiter=","

# Load the csv file using SQLContext
test_df = sqlsc.read.format(file_type)\
	.option("inferSchema",infer_schema)\
	.option("header",first_row_header)\
	.option("sep",delimiter)\
	.load(test_df_path)

col_names = test_df.schema.names
# remove the column that pandas placed in
col_names = [col_names[i] for i in range(len(col_names)) \
            if col_names[i] != '_c0']

test_df = test_df.select(*col_names)

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

# Testing dataframe for A and B vectors
test_df = vector_assemble_function(test_df, col_names_vector_a, 
				   outputCol='VectorA')

test_df = vector_assemble_function(test_df, col_names_vector_b, 
				   outputCol='VectorB')


# ---------------------------------------------------------------------
pb('Make predictions on both models')
# ---------------------------------------------------------------------

predictions_a = rf_model_a.transform(test_df)

predictions_b = rf_model_b.transform(test_df)


# ---------------------------------------------------------------------
pb('Show 5 predictions for all features (part a)')
# ---------------------------------------------------------------------

predictions_a.select('Prediction', 'Price', 'VectorA').show(5)

# ---------------------------------------------------------------------
pb('Show 5 predictions for Bathroom (part b)')
# ---------------------------------------------------------------------

predictions_b.select('Prediction', 'Price', 'VectorB').show(5)

# ---------------------------------------------------------------------
pb('Compute the root mean squared error')
# ---------------------------------------------------------------------

def compute_rmse(df):
	df = df.withColumn('difference', f.col('Price') - f.col('Prediction'))
	df = df.withColumn('squared_difference', f.pow(f.col('difference'), f.lit(2).astype(IntegerType())))
	rmse = df.select(f.sqrt(f.avg(f.col('squared_difference'))).alias('rmse'))
	return rmse


# ---------------------------------------------------------------------
pb('Show 5 predictions for all features (part a)')
# ---------------------------------------------------------------------

rmse_a = compute_rmse(predictions_a)
rmse_a.show()

# ---------------------------------------------------------------------
pb('Show rmse for predictions Bathroom feature (part b)')
# ---------------------------------------------------------------------

rmse_b = compute_rmse(predictions_b)
rmse_b.show()




