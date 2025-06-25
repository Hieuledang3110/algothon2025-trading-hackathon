#Eric is using this for testing, hoping to start afresh since the other test file has become a clusterfuck
 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


#Test function to check if the P/L feedback is working
def alwaysBuy(data):
    positions = [100000 for _ in range(50)]
    return positions

#Test function to check if the P/L feedback is working
def alwaysSell(data):
    positions = [-100000 for _ in range(50)]
    return positions