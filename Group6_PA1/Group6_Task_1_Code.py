# Group 6 Task 1 Code
# Tuesday, April 13, 2021

"""
<<Provide an overview of what is done in this script>>

bin/spark-submit Group6_Task_1_Code.py path/to/file.csv

"""


from pyspark import SparkConf, SparkContext
import sys
from pyspark.sql import SQLContext

# Give the spark configuration
conf = SparkConf().setMaster("local").setAppName("Group6Task1Code")
# Set the log level to "WARN" to hide all INFO lines printed out
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlsc = SQLContext(sc)

# Provide file information
file_location = sys.argv[1]  # get the file location from the command line arg
file_type = file_location.split('.')[-1]  # get the file type
if file_type != 'csv':  # if its not a csv warn that it should be
    raise ValueError('Expected a csv file.')

infer_schema="true"
first_row_header="true"
delimiter=","

# Load the text file using a relational database
RDD = sc.textFile(file_location) \
    .map(lambda line: line.split(",")) \
    .filter(lambda line: len(line)>1) \
    .map(lambda line: (line[0],line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10]))

print(RDD.take(10))

# Load the csv file using SQLContext
df = sqlsc.read.format(file_type)\
	.option("inferSchema",infer_schema)\
	.option("header",first_row_header)\
	.option("sep",delimiter)\
	.load(file_location)
# Get the number of rows in the dataframe
n_rows = df.count()
print('The number of rows in the dataset: {}'.format(n_rows))
# Get the number of columns in the dataframe
n_cols = len(df.columns)
print('The number of columns in the dataframe: {}'.format(n_cols))

df.printSchema()
