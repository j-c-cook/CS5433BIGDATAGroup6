# Task 1 Group 6 README

*Jack Cook*
*Govardhan Digumurthi*
*Thomas Okonkwo*

## Description of dataset

The spark program is used to determine the number of features (columns)
and rows that are in the dataset. The schema of the dataset is also 
displayed. The schema is an overview of the column names and the 
variable types that are in each column (or feature).  

![F2_DatasetDescription](Images/F2_Datadescription.png)

Figure 2: The number of rows and columns and the schema of the dataframe


At the very bottom of the data set (row 11,583) there exists a 
description of what some of the keys in the dataset mean:

- Suburb - name of suburb

- Type:
	- h - house, cottage, villa, semi, terrace
	- u - unit, duplex
	- t - townhouse site, development side
	- o - res, other residential
- Price: Price in dollars ($)
- Distance: Distance from CBD
- Zipcode: the zipcode where the unit is located
- Bedroom: number of bedrooms
- Bathroom: number of bathrooms
- Car Garage: number of car garages
- Lot Size: the size of the property (in acres)
- Region name: General region (West, North West, North, North East, etc.)
- Property count: Number of properties that exist in the suburb 

There are 7 region names, 3 housing types and 312 suburbs.

## Work done
There are rows at the bottom at the data that helped to describe the
data, but it did not contain any meaningful data for processing and has
been removed. 

Here is a list of the csv files and a description of what they are:

 -  Housing_data-Final-1.csv - the original csv supplied 
 -  Housing_data-Final-2.csv - the bottom 20 rows of garbage data are
    removed

## Instructions to run program

## Discussion of results
