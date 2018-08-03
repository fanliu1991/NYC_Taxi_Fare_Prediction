# NYC_Taxi_Fare_Prediction
Predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations

The evaluation metric for this project is the root mean-squared error (RMSE). 
RMSE measures the difference between the predictions of a model, and the corresponding ground truth.
A large RMSE is equivalent to a large average error, so smaller values of RMSE are better.
One nice property of RMSE is that the error is given in the units being measured, so you can tell very directly how incorrect the model might be on unseen data.


File descriptions:

train.csv - Input features and target fare_amount values for the training set (about 55M rows).
test.csv - Input features for the test set (about 10K rows). Project goal is to predict fare_amount for each row.
sample_submission.csv - a sample submission file in the correct format (columns key and fare_amount).
                        This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

Data fields:

key - Unique string identifying each row in both the training and test sets. 
      Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field. 

Features:

pickup_datetime - timestamp value indicating when the taxi ride started.
pickup_longitude - float for longitude coordinate of where the taxi ride started.
pickup_latitude - float for latitude coordinate of where the taxi ride started.
dropoff_longitude - float for longitude coordinate of where the taxi ride ended.
dropoff_latitude - float for latitude coordinate of where the taxi ride ended.
passenger_count - integer indicating the number of passengers in the taxi ride.

Target:

fare_amount - float dollar amount of the cost of the taxi ride. 
              This value is only in the training set; 
              This is what to be predicted in the test set.


