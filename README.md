# CycleCV


Bachelor Project by Mads, Marcus and Mikkel

Containing simulation of traffic at the widely discussed intersection at Dybbølsbro.

TODO: skriv how 2 execute scripts, og hvad de gør, format readme.md til at være gamer, https://dillinger.io/

## main functionality
1. create data directory
2. run wrangler.py --> pdf.pkl, nndf.pkl
3. run LSTM.py, LSTM_padding_aware.py, xgb_train.py --> model in models/
4. run simulation_script ????

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
| LSTM_padding_aware | create and train LSTM model on padded pdf.pkl, but removes padding in forward |
| ... | ... |
