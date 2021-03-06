# Task 2 Group 6 README

**Jack Cook**

**Govardhan Digumurthi**

**Thomas Okonkwo**

## Description of dataset and task to be done

In this task, we will use the output file - Group6_Task_1_Output.csv from task 1 to calculate
the correlation coefficients. The new dataset is a cleaned version of the original housing 
data. Here all the categorical/discrete data has been converted to indexed values for accurate
computation of the correlation coefficient across the dataset.

The new dataset has schema of the following:
```
 |-- Price: double (nullable = true)
 
 |-- Distance: double (nullable = true)
 
 |-- Zipcode: double (nullable = true)
 
 |-- #Bedroom: double (nullable = true)
 
 |-- #Bathroom: double (nullable = true)
 
 |-- #-Car Garage: double (nullable = true)
 
 |-- Lot_size: double (nullable = true)
 
 |-- Property_count: double (nullable = true)
 
 |-- Suburb_indexed: double (nullable = true)
 
 |-- Type_indexed: double (nullable = true)
 
 |-- Region_name_indexed: double (nullable = true)
```

The field names which has "indexed" appended to it are the field which were string values i.e
categorical/discrete values. They were converted and indexed so the computation of correlation
would be more accurate, instead of removing those fields.  

## Work done
Spark and pandas was used in this task.

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

Ideally, this task was done using spark and pandas. We used spark for Pearson correlation
method and pandas for Spearman and Kendall method. This is because Spark doesn't have all
correlation tools. So to calculate the correlation coefficient using spearman and kendall
we convert the spark dataframe to pandas, since pandas has the three correlation tools.

**Note: Pandas are not efficient for big data. I know we are not dealing with big data unfortunately, but spark is much better for big data.**

The following is a list of tasks performed in the code in order:
A spark configuration is defined and the Group6_Task1_Output.csv file 
is read into spark. Spark datafram has an inbuilt pearson correlation method. Since 
our correlation coefficient is based on predicting how the features in the dataset
affects the price of the house, we will be calculating the correlation of Price 
against all the housing data features.

In spark, we use the inbuilt pearson correlation method to run a computation accross all
the fields.

The remaining 2 methods of correlation coefficient, spark does not possess the tools needed
to compute the correlation using this method. So we convert the spark dataframe to pandas.

Hence we calculated the correlation coefficient using the spearman and kendall method
in pandas. 

## Instructions to run program
The code must be copied to the cluster:
```
scp Group6_Task_2_Code.py cookjc@hadoop-nn001.cs.okstate.edu:/home/cookjc
```

The python file must be executed using `spark-submit`:

```
spark-submit Group6_Task_2_Code.py Group6_Task_1_Output.csv
```

## Discussion of results
From the final result from the three correlation coefficient, we got different values
when we compute the correlation of price against all the other features of the housing 
data.

From the result of the computation of correlation coefficient we can deduce that across 
all the three methods for correllation, bedroom and bathroom  are the the major factors 
why price of a particular housing will be expensive. However we can also see that, the 
pearson correllation coefficient says otherwise. It says that the bathroom is responsible 
for increase in pricing unlike spearman and kendall method. 

Summarily, the diffence in values between the bedroom and bathroom accross all three
methods is really small. So it's safe to conclude that bedroom and bathroom are the major
factors which affects the pricing of an apartment or house.

Because the built in pyspark Person method indicates that `Bathroom` is
the most important feature, that is what we will move forward with. 
However, Spearman and Kendall indicating that Bedroom is more important
is an interesting finding. 

There are 6 files output from this task. Each method (Pearson, Kendall 
and Spearman) all have a csv and heatmap pdf exported. 
