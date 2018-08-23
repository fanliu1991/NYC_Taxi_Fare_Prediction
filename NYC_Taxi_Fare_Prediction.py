import os

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

import support_functions as sf
import background_information as bi

# read data in pandas dataframe
df_train =  pd.read_csv('../input/train.csv', nrows = 80_000, parse_dates=["pickup_datetime"])

# Data Cleaning

# check and drop the missing values
df_train = df_train.dropna(how = 'any', axis = 'rows')

# drop records that with negative fare amount
df_train = df_train[df_train.fare_amount>0]

# a taxi trip with zero passenger or more than 6 (a SUV cab) passengers is not realistic, so drop them
df_train = df_train[(df_train['passenger_count']>0) & (df_train['passenger_count']<=6)]

# trips that outside of New York City and its nearby areas are dropped
df_train = df_train[sf.city_nearby_areas(df_train, bi.nyc)]


# Features Extraction

# **Calculate the Distance In Kilometers Between Pickup and Dropoff Position**
df_train['distance_km'] = df_train.apply(lambda x: sf.distance(x["pickup_latitude"], x["pickup_longitude"], \
                                   x["dropoff_latitude"], x["dropoff_longitude"]), axis=1)
# check for outliers in distance 
print(df_train["distance_km"].describe())
# some trips have zero distance but non-zero fare
# maybe caused by pickup and dropoff at same place
# drop such trips since there can not supply valuable information to fit model
df_train = df_train[df_train["distance_km"] >= 0.1]

# **Check If a Taxi Trip is To/From Airport**
# taxi trip normally has metered fare, but with some exceptions
# taxi trip between JFK and Manhattan has a flat fare.
# taxi trip to EWR has a Newark Surcharge
df_train["to_airport"] = df_train.apply(lambda row: sf.airport_trip(row), axis=1)

# encoding to_airport feature by one-hot
df_airport = pd.get_dummies(df_train["to_airport"])
df_train = df_train.join(df_airport)


# **Extract Hour and Day From pickup_datetime**
# taxi trips in the night have extra surcharge.
# the amount of traffic also depends on the hour of the day, and it determines the duration of the trip and thus the fare.
df_train = df_train.apply(lambda r: extract_time(r), axis=1)

# encoding hour_type feature by one-hot
df_hour_type = pd.get_dummies(df_train["hour_type"])
df_train = df_train.join(df_hour_type)


# Check if a Taxi Trip is Group Ride**

# Two, three, or four people can take a group ride from a yellow taxi Group Ride Stand and pay a flat rate.
# Group Ride pickup times are certain hours, Monday – Friday (excluding holidays).

# therefore, in a group ride, fare amount supposed to be linearly dependent on the number of passengers,
# but such trend is not shown in the scatter plots of passenger_count v.s. fare_amount for any group ride
# thus, this feature is not a significant factor to influence fare amount


# **Check If Trips Crossed Boroughs, Which may Caused Toll Fees to Fare Amount**
# Passengers must pay tolls if the trip between New Jersey/Manhattan, 
# Manhattan/Queens+Brooklyn, Staten Island/Brooklyn, New Jersey/Staten Island
df_train["tolls_NJ_Manh"] = 0
df_train["tolls_Manh_BQ"] = 0
df_train["tolls_BQ_SI"] = 0
df_train["tolls_NJ_SI"] = 0
df_train = df_train.apply(lambda row: sf.tolls_fees(row), axis=1)
























