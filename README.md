# CS5433BIGDATAGroup6
Oklahoma State University CS 5433 BIG DATA final group project 
repository

## Task 1 Data correction
Housing data containing the following features:

- Suburb - name of suburb
- Type - the kind of unit (eg. house, townhouse, etc.)
- Price - price in dollars ($)
- Distance - distance from CBD
- Zipcode - the zipcode where the unit is located
- #Bedroom - the number of bedrooms in the unit
- #Bathroom - the number of bathrooms in the unit
- #Car Garage - the number of car garages
- Lot size - the size of the property (in acres)
- Region name - general region (west, north west, north, north east, 
  etc.)
- Property count - the number of properties that exist in the suburb

Some of the cells in the data contain out of range values, such as null
or values less than 0. Rather than removing the bad data, we implement
a cosine similarity algorithm to replace the bad values with values 
from the most similar row in the data set.
