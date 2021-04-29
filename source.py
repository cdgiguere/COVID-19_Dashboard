%matplotlib inline
from mpl_toolkits.basemap import Basemap
from google.colab import drive
import pandas as pd
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, Dropdown, IntSlider
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from contextlib import redirect_stdout
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
import sys
import warnings

drive.mount('/content/drive')

output_loc = '/content/drive/MyDrive/MITRE/Google_CoLab/Output/'

with open(output_loc + 'MASSgovCensusTown.csv') as f:
  df = pd.read_csv(f).dropna(how='all').dropna(axis=1, how='all')

warnings.filterwarnings('ignore')

# some variables to use throughout
counties = set(df['County'])
counties.add('All')

global topXTowns
topXTowns = None
global selectedCOVIDStat
selectedCOVIDStat = None
global selectedDemographicStat
selectedDemographicStat = None

distress = ['% Infection rate', '# Cases per 1000 citizens', '% Positive tests',
            '# Total positive tests', '# Total cases', '# Total cases last 2 weeks',
            '% Positive tests last 2 weeks']
demographic = ['% Population minority', '% Population no health insurance',
               '% Households w/ cohabiting couples', '% Population <18',
               '% Population >=18', '% Population >=21', '% Population >=65',
               '% Population w/ disablilty', '% Foreign-born not citizen',
               '% Population speaks english poorly',
               '% Households w/ internet', '% Population in poverty',
               '% Households no vehicle', '% Households incomplete plumbing',
               '% Households incomplete kitchen', '% Households no telephone',
               '% Educated w/ diploma or higher']
mask = ['% Population wears mask always', '% Population wears mask frequently',
        '% Population wears mask sometimes', '% Population wears mask rarely',
        '% Population wears mask never']

def doubleBarChart(input_df, bar1, bar2, title=None, y1_label=None, y2_label=None):
  if y1_label is None:
    y1_label = bar1
  if y2_label is None:
    y2_label = bar2
  if title is None:
    title = y1_label + ' VS ' + y2_label

  # linear regression
  X = input_df[bar1].values.reshape(-1, 1)
  Y = input_df[bar2].values.reshape(-1, 1)
  lin_reg = LinearRegression()
  lin_reg.fit(X, Y)
  Y_pred = lin_reg.predict(X)

  input_df["Y_pred"] = Y_pred
  ascending = [bar1 < bar2, False]
  df = pd.melt(input_df[['City/Town', bar1, bar2, "Y_pred"]],
              id_vars=["City/Town", 'Y_pred'], var_name="Metric",
              value_name="Value")
  df = df.sort_values(by=['Metric', 'Value'], ascending=ascending)

  # mask the smaller column by the mean of the larger. The y-axis will be
  # scaled back to display the correct values
  mask = df.Metric.isin([bar1])
  mask_1 = True
  if df[~mask]['Value'].mean() < df[mask]['Value'].mean():
    mask = df.Metric.isin([bar2])
    mask_1 = False
  scale = int(df[~mask]['Value'].mean() / df[mask]['Value'].mean())
  if scale != 0:
    df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * scale

  # create the graph
  sns.set(rc={'figure.figsize': (50, 10), 'axes.labelsize': 25,
              'ytick.labelsize': 17, 'axes.labelpad': 15, 'legend.fontsize': 20})
  fig, ax1 = plt.subplots()
  g = sns.barplot(x='City/Town', y="Value", hue="Metric", data=df, ax=ax1)
  g.set_title(title, fontsize=30)
  fig.autofmt_xdate()

  # create a second y-axis with the scaled ticks
  ax1.set_ylabel(y1_label)
  ax2 = ax1.twinx()

  # add R^2 to the legend
  blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
  r2 = lin_reg.score(X, Y)
  h, l = g.get_legend_handles_labels()
  g.legend(h + [blank], l + [f'R^2 score = {r2:.3f}'], title="Legend")

  # ensure ticks occur at the same positions, then modify labels
  if scale != 0:
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels(np.round(ax1.get_yticks() / scale, 4))
    ax2.set_ylabel(y2_label)

  # add regression line
  ax2.plot(range(len(X)), df.Y_pred[df['Metric'] == bar1]*scale, color='red')
  ax1.margins(x=0.00001)
  ax2.margins(x=0.00001)

  # change angle and fontsize of x-labels
  if len(df) <= 100:
    plt.setp(ax1.get_xticklabels(), rotation=45)
  elif len(df) <= 200:
    plt.setp(ax1.get_xticklabels(), rotation=60, size=15)
  elif len(df) <= 300:
    plt.setp(ax1.get_xticklabels(), rotation=75, size=13)
  else:
    plt.setp(ax1.get_xticklabels(), rotation=90, size=10)


def interactiveDoubleBarChart(feature1='% Population contracted virus', 
                              feature2='% Population minority', county='All'):
  cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])][1:]

  copy = df.copy()
  # County filter
  if county != 'All':
    copy = copy.loc[copy['County'] == county]

  x = min(copy[feature1])
  y = max(copy[feature1])
  u = min(copy[feature2])
  v = max(copy[feature2])
  n = len(copy)

  # Sliders
  def slider(x, y, u, v, copy):
    if y < x or v < u:
      print("Maximum can't be smaller than minimum")

    # Slider filters
    copy = copy[copy[feature1] >= x]
    copy = copy[copy[feature1] <= y]
    copy = copy[copy[feature2] >= u]
    copy = copy[copy[feature2] <= v]

    doubleBarChart(copy, feature1, feature2)

  interact_manual(slider, 
                  x=FloatSlider(min=x, max=y, step=((y-x)/n), description='Left Min:'),
                  y=FloatSlider(min=x, max=y, step=((y-x)/n), value=y, description='Left Max:'), 
                  u=FloatSlider(min=u, max=v, step=((v-u)/n), description='Right Min:'),
                  v=FloatSlider(min=u, max=v, step=((v-u)/n), value=v, description='Right Max:'), 
                  copy=fixed(copy))
  
  def singleBarChart(input_df, y, avg, title=None, y_label=None):
  if y_label is None:
    y_label = y
  if title is None:
    title = y_label + ' by town in MA'

  # sort by the feature to graph nicely
  df = input_df.sort_values(by=y, ascending=False)

  # create the graph
  sns.set(rc={'figure.figsize': (50, 10), 'axes.labelsize': 25,
              'ytick.labelsize': 17, 'axes.labelpad': 15, 'legend.fontsize': 20})
  fig, ax = plt.subplots()
  g = sns.barplot(x='City/Town', y=y, data=df, ax=ax)
  g.set_xticklabels(g.get_xticklabels(), fontsize = 25)
  g.set_title(title, fontsize=35)
  line = g.axhline(avg, color='red')
  plt.legend(handles=[line], labels=['Avg. ' + y_label + ' for all of MA'])
  fig.autofmt_xdate()

  # change angle and fontsize of x-labels
  if len(df) <= 100:
    plt.setp(ax.get_xticklabels(), rotation=45)
  elif len(df) <= 200:
    plt.setp(ax.get_xticklabels(), rotation=60, size=15)
  elif len(df) <= 300:
    plt.setp(ax.get_xticklabels(), rotation=75, size=13)
  else:
    plt.setp(ax.get_xticklabels(), rotation=90, size=10)


def interactiveSingleBarChart(feature='% Population contracted virus', county='All'):
  cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])][1:]

  copy = df.copy()
  avg = df[feature].mean()
  # County filter
  if county != 'All':
    copy = copy.loc[copy['County'] == county]

  x = min(copy[feature])
  y = max(copy[feature])
  n = len(copy)

  # Sliders
  def slider(x, y, copy):
    if y <= x:
      print("Maximum must be larger than minimum")
      sys.exit(0)

    # Slider filters
    copy = copy[copy[feature] >= x]
    copy = copy[copy[feature] <= y]

    # create the bargraph
    singleBarChart(copy, feature, avg)

    # set global variables to keep track of what they are looking at
    global topXTowns
    topXTowns = list(copy['City/Town'])
    global selectedCOVIDStat
    selectedCOVIDStat = feature

  interact_manual(slider, 
                  x=FloatSlider(min=x, max=y, value=12.6, step=((y-x)/n), description='Min:'),
                  y=FloatSlider(min=x, max=y, value=y, step=((y-x)/n), description='Max:'),  
                  copy=fixed(copy), avg=fixed(avg))
  
def bubbleMap(copy, size, diffTowns=False, green='None', blue='None', scale=5, setDem=False):
  if setDem:
    # make a towns of interest column
    copy['Interest'] = [1 if town in topXTowns else 0 for town in copy['City/Town']]
  
  # set the map location and fetch the background
  fig = plt.figure(figsize=(30, 30))
  map = Basemap(llcrnrlat=41.2000, llcrnrlon=-73.5100,
                urcrnrlat=42.8900, urcrnrlon=-69.8600,
                epsg=4269, resolution='l')
  map.arcgisimage(service='World_Street_Map', xpixels = 1500)

  # fill in borders
  map.drawstates(color='black')

  # plot city points for the chosen colors
  colors = {green: 'green', blue: 'blue', size: 'red'}
  order = [col for col in [size, green, blue] if col != 'None']
  if diffTowns:
    map.scatter(list(copy['Longitude']), list(copy['Latitude']), latlon=True,
                c=list(copy['Interest']), s=list(copy[size]*scale), alpha=0.5,
                cmap='vlag')
  else:
    order.sort(key=lambda x: df[x].mean(), reverse=True)
    for i, ord in enumerate(order):
      map.scatter(list(copy['Longitude']), list(copy['Latitude']), latlon=True,
                  c=colors[ord], s=list((copy[ord]*scale*(3-i))), alpha=0.5)

  # label the top X towns in the legend
  if diffTowns:
    plt.scatter([], [], c='red', s=500, alpha=0.5, label='Top ' + str(len(topXTowns)) + ' towns for ' + size)
    plt.scatter([], [], c='blue', s=500, alpha=0.5, label='Other towns')
  # or label the different features for different colors
  else:
    for a in order:
      plt.scatter([], [], c=colors[a], s=500, alpha=0.5, label=a)

  plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right', fontsize=15);
  plt.show()

  if setDem:
    global selectedDemographicStat
    selectedDemographicStat = size
