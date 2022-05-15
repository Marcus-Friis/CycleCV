# CycleCV


Bachelor Project by Mads Høgenhaug, Marcus Friis and Mikkel Petersen.

This repository contains all code used for wrangling data, training models and simulating traffic, along with other extra scripts for visualising data and more. 

## main pipeline
To run the core functionality, the core flow is
1. run wrangler.py --> pdf.pkl, nndf.pkl
2. run train_test_split.py --> split pdf.pkl and nndf.pkl in tran-test split
3. run LSTM.py, model_train.py --> models
4. run simulate_results.py --> multiple videos of simulations

## visualize data in video
To sync source video with data, do
1. create frames directory
2. run video_to_frames.py --> framei.jpg in frames/
3. run animate_data.py --> data_animation.mp4

## script overview
| FILE | DESCRIPTION |
| ---- | ----------- |
| animate_data.py | overlay data with video |
| clustering_helper_funcs.py | helper functions for clustering with HDBScan |
| LSTM.py | create and train LSTM model on padded pdf.pkl |
| LSTM_padding_aware.py | create and train LSTM model on padded pdf.pkl, but removes padding in forward, test script |
| model_train.py | script trains and tests gradient boosted models and sklearn models |
| pytrajectory.py | trajectory class for pytorch models |
| shape_helper_funcs.py | helper functions for using shapely for zone creation |
| simulate_results.py | simulates intersection using previously trained models |
| sktrajectory.py | trajectory class for models with sklearn interface |
| train_test_split.py | splits pdf.pkl and nndf.pkl into train and test split |
| trajectory.py | class for handling internals of trajectories when simulating |
| video_to_frames.py | converts every frame of video to .jpg |
| wrangler.py | this script extracts all features and handles general data wrangling |



## dependencies
This project was created using python version 3.8.5 with the following packages and versions.
| PACKAGE | VERSION |
| ------- | ------- |
| catboost | 1.0.5 |
| hdbscan | 0.8.27 |
| lightgbm | 3.3.2 |
| matplotlib | 3.3.2 |
| numpy | 1.19.2 |
| opencv-python | 4.5.5.64 |
| pandas | 1.4.1 |
| pillow | 8.0.1 |
| scikit-learn | 0.24.2 |
| seaborn | 0.11.0 |
| shapely | 1.8.1.post1 |
| torch | 1.11.0 |
| xgboost | 1.6.0 |
