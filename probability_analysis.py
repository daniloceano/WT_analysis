#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:50:18 2021

@author: danilocoutodsouza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

file = '../all_WT_dates2.txt'
data = pd.read_csv(file) 
time = np.arange(1979,2011)
WT = np.arange(1,37)