import pickle
import pandas as pd
import numpy as np
import configparser

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns

from mikkel import *

import hdbscan

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from wrangler import Wrangler
from clustering_helper_funcs import *


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pytorch import *


def main():
	config = configparser.ConfigParser()
	config.read('settings.ini')
	config = config['DEFAULT']

	wrangle_bool = config.getboolean('wrangle')
	if wrangle_bool:
		# load trajectory dataframes from pickle and merge to get frame data
		print('loading data...')
		df = Wrangler.load_pickle('bsc-3m/traj_01_elab.pkl')
		df_frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
		df = df.join(df_frames['frames'])

		# load traffic lights coordinates and color info
		l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')
		l_df = pd.read_csv('bsc-3m/signals_dense.csv')

		# select strictly cars, remove later?
		df, _ = Wrangler.filter_class(df, ['Car'])

		# cluster and remove outliers
		# HDBSCAN for now, try other in future?
		print('clustering...')
		min_cluster_size, min_samples, cluster_selection_epsilon = get_hyperparameters('Car', '')
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			cluster_selection_epsilon=cluster_selection_epsilon
		)
		x = df[['x0', 'y0', 'x1', 'y1']].to_numpy()  # prepare data for clustering
		xc = np.array(clusterer.fit_predict(x))
		df['cluster'] = xc
		fdf = detect_outliers(clusterer, df)
		fdf = fdf.loc[fdf['cluster'] != -1]

		# wrangle data into shape
		print('wrangling data...')
		wr = Wrangler(fdf, l_xy, l_df)\
			.init_attributes(step_size=config.getint('step_size'), dump=config.getboolean('dump_data'), path=config['data_path']+'pdf.pkl')\
			.get_nndf(dump=config.getboolean('dump_data'), path=config['data_path']+'nndf.pkl')
		nndf = wr.nndf

	else:
		pdf = Wrangler.load_pickle(config['data_path'] + 'pdf.pkl')
		nndf = Wrangler.load_pickle(config['data_path'] + 'nndf.pkl')

	plot_bool = config.getboolean('plot')
	if plot_bool:
		pass

	load_model = config.getboolean('load_model')
	if not load_model:
		# rewrite split to use whole trajectories
		print('preparing data...')
		cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
		        'l2', 'l3', 'dir_0', 'dir_1', 'dir_2', 'target']
		df_train, df_val = train_test_split(nndf[cols], test_size=0.2, random_state=1)
		df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=1)


		x_train, y_train = df_train[cols[:-1]].to_numpy(), df_train[cols[-1:]].to_numpy().reshape(-1)
		x_val, y_val = df_val[cols[:-1]].to_numpy(), df_val[cols[-1:]].to_numpy().reshape(-1)
		x_test, y_test = df_test[cols[:-1]].to_numpy(), df_test[cols[-1:]].to_numpy().reshape(-1)

		x_train,y_train = torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float().unsqueeze(-1)
		x_val,y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float().unsqueeze(-1)
		x_test,y_test = torch.from_numpy(x_test).float(),torch.from_numpy(y_test).float().unsqueeze(-1)

		train = TensorDataset(x_train,y_train)
		train_loader = DataLoader(train,batch_size=64,shuffle = False, drop_last=True)

		val = TensorDataset(x_val,y_val)
		val_loader= DataLoader(val,batch_size=64,shuffle=False,drop_last=True)

		test = TensorDataset(x_test,y_test)
		test_loader = DataLoader(test,batch_size=64,shuffle=False,drop_last=True)
		test_loader_one = DataLoader(test,batch_size=1,shuffle=False,drop_last=True)

		cuda = torch.cuda.is_available()
		device = "cpu"




		#hyper parameters
		# input_dim = 8
		#output_dim = 2
		#hidden_dim = 100
		#layer_dim = 3
		batch_size = 64
		dropout = 0.2
		n_epochs = 1
		learning_rate = 1e-3
		weight_decay = 1e-6

		# model_params = {'input_dim': input_dim,
		#                 'hidden_dim' : hidden_dim,
		#                 'layer_dim' : layer_dim,
		#                 'output_dim' : output_dim,
		#                 'dropout_prob' : dropout}

		model = Mikkel()#get_model('LSTM', model_params).to(device)


		loss_fn = nn.MSELoss(reduction="mean")
		optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

		print('training model...')
		opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
		print("opt done")
		opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=13)
		print("opt train done")
		opt.plot_losses()


		print("calculating predictions and values")
		predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=13)

		#print("Prediction:\t", predictions, "\n", "Values:\t",values)

		dump_model = config.getboolean('dump_model')
		if dump_model:
			model_str = 'model.pkl'
			torch.save(model.state_dict(), config['model_path']+model_str)


		else:
			model_str = 'model.pkl'
			clf = Wrangler.load_pickle(config['model_path']+model_str)

		simulate = config.getboolean('simulate')
		if simulate:
			pass

if __name__ == '__main__':
	main()
