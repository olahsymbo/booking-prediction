## Booking Prediction

The aim of this project is to predict whether an online user will make a booking or not based on the online activities of the users.


### Details

Because the non-booker class is less than booker class we used resampling method to undersample the booker class. Afterward we filter out features with high correlation. And for training Gradient Boosting Classifier (GBC) is used. 

In order to perform hyperparameter selection, we used k-fold cross validation to determine the best `max_depth`, `n_estimators`, `min_samples_leaf`.


### Running the codes:


- to train models again, cd into directory and run 

  ```
  python3 training_file.py
  ```
 
- to make predictions run.

  ```
  python3 testing_file.py
  ```
