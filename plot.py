
import numpy as np

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import colors

import pandas as pd

import seaborn as sns

#Use tex for labelling the plots
plt.rcParams['text.usetex'] = True

def plot_CI(values:np.array,
	xlabel,
	ylabel,
	filename):
	fig, ax = plt.subplots()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	num_rows = values.shape[0]
	num_columns = values.shape[1]

	episodes = np.zeros(num_columns)

	mean = np.zeros(num_columns)
	stdev = np.zeros(num_columns)

	for c in range(0, num_columns):
		mean[c] = np.mean(values[:,c])
		stdev[c] = np.std(values[:,c])
		episodes[c] = c


	lower_curve = mean - 2*stdev
	upper_curve = mean + 2*stdev

	plt.plot(mean)
	plt.fill_between(episodes, lower_curve, upper_curve, color='b')
	plt.savefig(filename)
	plt.clf()

def compare_plot_CI(values1: np.array,
	label1,
	values2:np.array,
	label2,
	xlabel,
	ylabel,
	filename,
	pfailgraph = False):
	fig, ax = plt.subplots()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	num_rows = values1.shape[0]
	num_columns = values1.shape[1]

	episodes = np.zeros(num_columns)

	mean = np.zeros(num_columns)
	stdev = np.zeros(num_columns)

	for c in range(0, num_columns):
		mean[c] = np.mean(values1[:,c])
		stdev[c] = np.std(values1[:,c])
		episodes[c] = c

	lower_curve = mean - 2*stdev
	upper_curve = mean + 2*stdev

	plt.plot(mean, label = label1)
	plt.fill_between(episodes, lower_curve, upper_curve, color='b', alpha = 0.1)

	num_rows = values2.shape[0]
	num_columns = values2.shape[1]

	episodes = np.zeros(num_columns)

	mean = np.zeros(num_columns)
	stdev = np.zeros(num_columns)

	for c in range(0, num_columns):
		mean[c] = np.mean(values2[:,c])
		stdev[c] = np.std(values2[:,c])
		episodes[c] = c


	lower_curve = mean - 2*stdev
	upper_curve = mean + 2*stdev

	plt.plot(mean, label = label2)
	plt.fill_between(episodes, lower_curve, upper_curve, color='r', alpha = 0.1)
	if pfailgraph:
		plt.ylim(0, 1)
	plt.legend(loc="upper right")
	plt.savefig(filename)
	plt.clf()



def compare_plot_CI_seaborn(values1: np.array,
	label1,
	values2:np.array,
	label2,
	xlabel,
	ylabel,
	filename,
	pfailgraph = False):

	values1_df = pd.DataFrame(values1)
	values2_df = pd.DataFrame(values2)

	values1_df = pd.melt(frame = values1_df,
	             var_name = 'Episodes',
	             value_name = 'runs')

	values2_df = pd.melt(frame = values2_df,
	             var_name = 'Episodes',
	             value_name = 'runs')

	fig, ax = plt.subplots()
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	
	sns.lineplot(ax = ax,
	             data = values1_df,
	             x = 'Episodes',
	             y = 'runs',
	             label = label1)
	print("Completed first CI evaluation")
	sns.lineplot(ax = ax,
	             data = values2_df,
	             x = 'Episodes',
	             y = 'runs',
	             label = label2)
	print("Completed second CI evaluation")
	ax.legend(loc="upper right")
	if pfailgraph:
		plt.ylim(0, 1)
	plt.savefig(filename)
	plt.clf()


