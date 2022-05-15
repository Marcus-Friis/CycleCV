# CycleCV


Bachelor Project by Mads, Marcus and Mikkel

Containing simulation of traffic at the widely discussed intersection at Dybbølsbro.

## main pipeline
1. create data directory
2. run wrangler.py --> pdf.pkl, nndf.pkl
3. run train_test_split.py --> split pdf.pkl and nndf.pkl in tran-test split
4. run LSTM.py, model_train.py --> model in models/
5. run simulate_results.py --> multiple videos of simulations

## visualize data in video
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
