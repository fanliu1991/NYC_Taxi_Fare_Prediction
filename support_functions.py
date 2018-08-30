import os

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path
from sklearn.metrics import mean_squared_error, explained_variance_score

import background_information as bi


# The valid range of latitudes and longitudes is as following:
# Latitudes range from -90 to 90.
# Longitudes range from -180 to 180.

# Check if pickup and dropoff positions are located in the city and nearby area
def city_nearby_areas(df, city):
    return (df['pickup_longitude'] >= city[1]-1.5) & (df['pickup_longitude'] <= city[1]+1.5) & \
           (df['pickup_latitude'] >= city[0]-1.5) & (df['pickup_latitude'] <= city[0]+1.5) & \
           (df['dropoff_longitude'] >= city[1]-1.5) & (df['dropoff_longitude'] <= city[1]+1.5) & \
           (df['dropoff_latitude'] >= city[0]-1.5) & (df['dropoff_latitude'] <= city[0]+1.5)


# calculate distance between two latitude longitude points.
# this formula is based on https://stackoverflow.com/questions/27928/
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...


# check if a give place (coordinate) is loacted in the specified area
def in_area(area, place):
    p = path.Path(area)
    res = p.contains_point(place)
    return res

# check if a trip is between JFK and Manhattan, or a trip is to Newark Airport,
# dropoff/pickup at less than 2 km from airport is regarded as trip to/from airport
def airport_trip(trip):
    if distance(trip["dropoff_latitude"], trip["dropoff_longitude"], bi.jfk[0], bi.jfk[1]) <= 2 and \
    in_area(bi.Manhattan, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True:
        return "jfk_Manhattan"
    elif distance(trip["pickup_latitude"], trip["pickup_longitude"], bi.jfk[0], bi.jfk[1]) <= 2 and \
    in_area(bi.Manhattan, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        return "jfk_Manhattan"
    elif distance(trip["dropoff_latitude"], trip["dropoff_longitude"], bi.ewr[0], bi.ewr[1]) <=2:
        return "airport_ewr"
    else:
        return "metered_fare"

# extract date and time details of taxi trips, determine when a taxi trip occured in a day
def hour_type(hour, weekday):
    if hour in list(range(20, 25)) + list(range(0,7)):
        return "night"
    elif hour in range(7, 10) and weekday == 1:
        return "morning_peak"
    elif hour in range(16, 20) and weekday == 1:
        return "afternoon_peak"
    else:
        return "normal_hour"

def extract_time(record):
    record['year'] = record["pickup_datetime"].year
    record['month'] = record["pickup_datetime"].month
    record["day_of_week"] = record["pickup_datetime"].weekday()
    record["weekday"] = 1 if record["day_of_week"] in range(0, 5) else 0
    record['hour'] = record["pickup_datetime"].hour
    record["hour_type"] = hour_type(record["hour"], record["weekday"])
    return record

# determine if tolls happened in a taxi trip
def tolls_fees(trip):
    # trip pickup at New Jersey
    if in_area(bi.New_Jersey, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Manhattan, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_NJ_Manh"] = 1
    elif in_area(bi.New_Jersey, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Brooklyn_Queens, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_NJ_Manh"] = 1
        trip["tolls_Manh_BQ"] = 1
    elif in_area(bi.New_Jersey, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Staten_Island, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_NJ_SI"] = 1
    
    # trip pickup at Manhattan
    elif in_area(bi.Manhattan, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Brooklyn_Queens, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_Manh_BQ"] = 1
    elif in_area(bi.Manhattan, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Staten_Island, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_BQ_SI"] = 1
    # no tolls fee from Manhattan to New Jersey
    
    # trip pickup at Brooklyn or Queens
    elif in_area(bi.Brooklyn_Queens, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.New_Jersey, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_Manh_BQ"] = 1
    elif in_area(bi.Brooklyn_Queens, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Manhattan, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_Manh_BQ"] = 1
    elif in_area(bi.Brooklyn_Queens, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Staten_Island, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_BQ_SI"] = 1
    
    # trip pickup at Staten Island
    elif in_area(bi.Staten_Island, [trip["pickup_latitude"], trip["pickup_longitude"]]) == True and \
    in_area(bi.Manhattan, [trip["dropoff_latitude"], trip["dropoff_longitude"]]) == True:
        trip["tolls_Manh_BQ"] = 1
    # no tolls fee from Staten Island to New Jersey, Brooklyn and Queens
    
    return trip

# calculate fare amount prediction based on features and coefficients
def predict_fare_amount(row, coef_dic):
    input_value = [1] + row.tolist()
    coef = coef_dic[row["year"]]
    fare_amount_prediction = sum([a*b for a,b in zip(input_value, coef)]).round(decimals = 2)
    return fare_amount_prediction

# make a plot to show the difference between real fare amount and predicted fare amount
def plot_prediction_analysis(real_fare, predicted_fare):
    plt.scatter(real_fare, predicted_fare)
    plt.xlabel("real fare amount")
    plt.ylabel("predicted fare amount")
    # the largest fare amount is
    max_fare = max(list(real_fare) + list(predicted_fare))
    plt.plot([0, max_fare], [0, max_fare], color='red', linestyle='-', linewidth=0.5)
    rmse = np.sqrt(mean_squared_error(real_fare, predicted_fare))
    plt.title('rmse = {:.2f}, with zero residual diagonal line'.format(rmse))
    plt.show()


# preprocessing dataset to extract features
def preprocess_dataset(df):
    df['distance_km'] = df.apply(lambda x: distance(x["pickup_latitude"], x["pickup_longitude"], \
                                   x["dropoff_latitude"], x["dropoff_longitude"]), axis=1)
    df["airport"] = df.apply(lambda x: to_airport(x["dropoff_latitude"], x["dropoff_longitude"]), axis=1)
    df = df.apply(lambda r: extract_time(r), axis=1)
    df_hour_type = pd.get_dummies(df["hour_type"])
    df = df.join(df_hour_type)
    df["fare_amount_no_surage"] = df.apply(lambda r: remove_surage(r), axis=1)
    return df



