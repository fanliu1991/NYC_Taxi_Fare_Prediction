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

# **Check if a Taxi Trip is Group Ride**
# Two, three, or four people can take a group ride from a yellow taxi Group Ride Stand and pay a flat rate.
# Group Ride pickup times are certain hours, Monday - Friday (excluding holidays).

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


# Model Fitting
# Considered inflation, models are fitted for each year.

# for metered fare trip: 
# fare_amount ~ distance_km + passenger_count + weekday + airport_ewr + afternoon_peak + morning_peak + night + normal_hour + tolls_NJ_Manh + tolls_BQ_SI + tolls_NJ_SI + tolls_Manh_BQ
   
# for flat fare trip, whick not depends on the distance:
# fare_amount ~ passenger_count + weekday + afternoon_peak + morning_peak + night + normal_hour + tolls_NJ_Manh + tolls_BQ_SI + tolls_NJ_SI + tolls_Manh_BQ

normal_trip_features = ["distance_km", "passenger_count", "weekday", "airport_ewr", \
                         "afternoon_peak", "morning_peak", "night", "normal_hour", \
                         "tolls_NJ_Manh", "tolls_BQ_SI", "tolls_NJ_SI", "tolls_Manh_BQ", "year"]

jfk_Manhattan_features = ["passenger_count", "weekday", "afternoon_peak", "morning_peak", "night", "normal_hour", \
                      "tolls_NJ_Manh", "year"]
 
# separate trips into two groups and fit seperate models for them
df_train_normal_trip = df_train[df_train["to_airport"] != "jfk_Manhattan"]
df_train_jfk_Manhattan = df_train[df_train['to_airport'] == 'jfk_Manhattan']

# seperate dataset to training dataset and validation dataset
normal_trip_X = df_train_normal_trip[normal_trip_features]
normal_trip_y = df_train_normal_trip[['fare_amount', 'year']]
normal_trip_X_train, normal_trip_X_validate, normal_trip_y_train, normal_trip_y_validate = train_test_split(normal_trip_X, normal_trip_y, test_size=0.25)

jfk_Manhattan_X = df_train_jfk_Manhattan[jfk_Manhattan_features]
jfk_Manhattan_y = df_train_jfk_Manhattan[['fare_amount', 'year']]
jfk_Manhattan_X_train, jfk_Manhattan_X_validate, jfk_Manhattan_y_train, jfk_Manhattan_y_validate = train_test_split(jfk_Manhattan_X, jfk_Manhattan_y, test_size=0.25)

# fit regression models
metered_fare_coef_dic = {}
flat_fare_coef_dic = {}

for year in list(df_train['year'].unique()):
    normal_trip_features = normal_trip_X_train[normal_trip_X_train['year'] == year].drop(columns=['year'])
    normal_trip_fare = normal_trip_y_train[normal_trip_y_train['year'] == year].drop(columns=['year'])
    normal_trip_lm = LinearRegression()
    normal_trip_lm.fit(normal_trip_features, normal_trip_fare)
    intercept = normal_trip_lm.intercept_
    coef = normal_trip_lm.coef_
    metered_fare_coef_dic[year] = np.concatenate((intercept, coef), axis=None)
    
    jfk_Manhattan_features = jfk_Manhattan_X_train[jfk_Manhattan_X_train['year'] == year].drop(columns=['year'])
    jfk_Manhattan_fare = jfk_Manhattan_y_train[jfk_Manhattan_y_train['year'] == year].drop(columns=['year'])
    jfk_Manhattan_lm = LinearRegression()
    jfk_Manhattan_lm.fit(jfk_Manhattan_features, jfk_Manhattan_fare)
    intercept = jfk_Manhattan_lm.intercept_
    coef = jfk_Manhattan_lm.coef_
    flat_fare_coef_dic[year] = np.concatenate((intercept, coef), axis=None)

# predict taxi fare amount for training dataset and validation dataset
normal_trip_X_train["prediction"] = normal_trip_X_train.apply(lambda row: sf.predict_fare_amount(row, "normal"), axis=1)
normal_trip_X_validate["prediction"] = normal_trip_X_validate.apply(lambda row: sf.predict_fare_amount(row, "normal"), axis=1)

jfk_Manhattan_X_train["prediction"] = jfk_Manhattan_X_train.apply(lambda row: sf.predict_fare_amount(row, "JFK"), axis=1)
jfk_Manhattan_X_validate["prediction"] = jfk_Manhattan_X_validate.apply(lambda row: sf.predict_fare_amount(row, "JFK"), axis=1)

# prediction plot for training dataset of normal trip
sf.plot_prediction_analysis(normal_trip_y_train["fare_amount"], normal_trip_X_train["prediction"])
# prediction plot for validation dataset of normal trip
sf.plot_prediction_analysis(normal_trip_y_validate["fare_amount"], normal_trip_X_validate["prediction"])

# prediction plot for training dataset of trip between JFK and Manhattan
sf.plot_prediction_analysis(jfk_Manhattan_y_train["fare_amount"], jfk_Manhattan_X_train["prediction"])
# prediction plot for validation dataset of trip between JFK and Manhattan
sf.plot_prediction_analysis(jfk_Manhattan_y_validate["fare_amount"], jfk_Manhattan_X_validate["prediction"])


#**Test Dataset Experiment**
df_test = pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])








