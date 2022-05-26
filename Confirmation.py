# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 02:41:03 2021

Preface:
    All angles such as roll, pitch ,or yaw are assumed to be in radians
@author: Bailey K. Srimoungchanh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from sympy import sin, cos, pi, atan2, sqrt
from pathlib import Path

# Helper function to put together the FPR/TPR CSVs generated for LPF testing
def combine_FPR_TPR_csv(path):
    tested_values = [2,   4,   6,   8,  10,  12,  14,  16,  18,  20,
            22,  24,  26,  28,  30,  32,  34,  36,  38,  40,  42,
            44,  46,  48,  50,  52,  54,  56,  57,  60,  62,  64,
            66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,
            88,  90,  92,  94,  96,  98, 100]
    
    BenignFiles = listdir(path + 'Results/Benign/PairwiseData/alpha/')
    AttackFiles = listdir(path + 'Results/Attack/PairwiseData/alpha/')
    
    #Processing the Benign files first
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Benign/PairwiseData/alpha/' + BenignFiles[0])
    for i in range(len(Missions)):
        temp = Missions[i].split('-')
        # Benign missions will always be format of "C||P-MissionName.csv"
        Missions[i] = "-".join(temp[:2])
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        Path(path + "Results/Benign/PairwiseData/Combined/" + Missions[i]).mkdir(parents=True, exist_ok=True)
        
        # Setting the columns to 2,4,6,..,100
        GPSOF_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUOF_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUGPS_CSV = pd.DataFrame(columns=range(2,102,2))
        
        # Populating them with dummy values for rows 1,2,...,50
        for j in range(2,102,2):
            GPSOF_CSV[j] =  [0.0] * 50
            IMUOF_CSV[j] = [0.0] * 50
            IMUGPS_CSV[j] = [0.0] * 50
        
        # Replacing row index with 2,4,...,100
        GPSOF_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUOF_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUGPS_CSV.index = pd.RangeIndex(2, 102, 2)
        
        # During testing of 2,4,6,...,100, 58 got replaced with 57. Need to replace
        # here to stay consistent
        GPSOF_CSV.rename({58:57},axis=0,inplace=True)
        IMUOF_CSV.rename({58:57},axis=0,inplace=True)
        IMUGPS_CSV.rename({58:57},axis=0,inplace=True)
        GPSOF_CSV.rename({58:57},axis=1,inplace=True)
        IMUOF_CSV.rename({58:57},axis=1,inplace=True)
        IMUGPS_CSV.rename({58:57},axis=1,inplace=True)
        
        for imu in tested_values:
            for of in tested_values:
                files = listdir(path + 'Results/Benign/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of))
                
                # Only consolidating the Net results, matching on mission name and net
                target = [match for match in files if Missions[i] in match and 'Net' in match]
                temp = pd.read_csv(path + 'Results/Benign/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of) + "/" + target[0])
                
                # Assign the data to a row,column named for the LPF value,
                # Rows are the OF LPF value, Columns are the ACC LPF value
                GPSOF_CSV.at[of,imu] = temp['GPSOF(FPR)'][0]
                IMUOF_CSV.at[of,imu] = temp['ACCOF(FPR)'][0]
                IMUGPS_CSV.at[of,imu] = temp['ACCGPS(FPR)'][0]
        
        GPSOF_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/GPSOF.csv',index=True)
        IMUOF_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/IMUOF.csv',index=True)
        IMUGPS_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/IMUGPS.csv',index=True)
    
    #Processing the Attack files second
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Attack/PairwiseData/alpha/' + AttackFiles[0])
    for i in range(len(Missions)):
        temp = Missions[i].split('-')
        # Attack missions will always be format of "C||P-MissionName-Sensor.csv"
        Missions[i] = "-".join(temp[:3])
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        Path(path + "Results/Attack/PairwiseData/Combined/" + Missions[i]).mkdir(parents=True, exist_ok=True)
        
        # Setting the columns to 2,4,6,..,100
        GPSOF_FPR_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUOF_FPR_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUGPS_FPR_CSV = pd.DataFrame(columns=range(2,102,2))
        GPSOF_TPR_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUOF_TPR_CSV = pd.DataFrame(columns=range(2,102,2))
        IMUGPS_TPR_CSV = pd.DataFrame(columns=range(2,102,2))
        
        # Populating them with dummy values for rows 1,2,...,50
        for j in range(2,102,2):
            GPSOF_FPR_CSV[j] =  [0.0] * 50
            IMUOF_FPR_CSV[j] = [0.0] * 50
            IMUGPS_FPR_CSV[j] = [0.0] * 50
            GPSOF_TPR_CSV[j] =  [0.0] * 50
            IMUOF_TPR_CSV[j] = [0.0] * 50
            IMUGPS_TPR_CSV[j] = [0.0] * 50
        
        # Replacing row index with 2,4,...,100
        GPSOF_FPR_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUOF_FPR_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUGPS_FPR_CSV.index = pd.RangeIndex(2, 102, 2)
        GPSOF_TPR_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUOF_TPR_CSV.index = pd.RangeIndex(2, 102, 2)
        IMUGPS_TPR_CSV.index = pd.RangeIndex(2, 102, 2)
        
        # During testing of 2,4,6,...,100, 58 got replaced with 57. Need to replace
        # here to stay consistent
        GPSOF_FPR_CSV.rename({58:57},axis=0,inplace=True)
        IMUOF_FPR_CSV.rename({58:57},axis=0,inplace=True)
        IMUGPS_FPR_CSV.rename({58:57},axis=0,inplace=True)
        GPSOF_FPR_CSV.rename({58:57},axis=1,inplace=True)
        IMUOF_FPR_CSV.rename({58:57},axis=1,inplace=True)
        IMUGPS_FPR_CSV.rename({58:57},axis=1,inplace=True)
        GPSOF_TPR_CSV.rename({58:57},axis=0,inplace=True)
        IMUOF_TPR_CSV.rename({58:57},axis=0,inplace=True)
        IMUGPS_TPR_CSV.rename({58:57},axis=0,inplace=True)
        GPSOF_TPR_CSV.rename({58:57},axis=1,inplace=True)
        IMUOF_TPR_CSV.rename({58:57},axis=1,inplace=True)
        IMUGPS_TPR_CSV.rename({58:57},axis=1,inplace=True)
        
        for imu in tested_values:
            for of in tested_values:
                files = listdir(path + 'Results/Attack/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of))
            
                # Only consolidating the Net results, matching on mission name and net
                target = [match for match in files if Missions[i] in match and 'Net' in match]
                temp = pd.read_csv(path + 'Results/Attack/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of) + "/" + target[0])
           
                # Assign the data to a row,column named for the LPF value,
                # Rows are the OF LPF value, Columns are the ACC LPF value
                GPSOF_FPR_CSV.at[of, imu] = temp['GPSOF(FPR)'][0]
                IMUOF_FPR_CSV.at[of,imu] = temp['ACCOF(FPR)'][0]
                IMUGPS_FPR_CSV.at[of,imu] = temp['ACCGPS(FPR)'][0]
                GPSOF_TPR_CSV.at[of,imu] = temp['GPSOF(TPR)'][0]
                IMUOF_TPR_CSV.at[of,imu] = temp['ACCOF(TPR)'][0]
                IMUGPS_TPR_CSV.at[of,imu] = temp['ACCGPS(TPR)'][0]
        
        GPSOF_FPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/GPSOF_FPR.csv',index=True)
        IMUOF_FPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUOF_FPR.csv',index=True)
        IMUGPS_FPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUGPS_FPR.csv',index=True)
        GPSOF_TPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/GPSOF_TPR.csv',index=True)
        IMUOF_TPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUOF_TPR.csv',index=True)
        IMUGPS_TPR_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUGPS_TPR.csv',index=True)
        
def combine_system_TTD_csv(path):

    tested_values = [2,   4,   6,   8,  10,  12,  14,  16,  18,  20,
            22,  24,  26,  28,  30,  32,  34,  36,  38,  40,  42,
            44,  46,  48,  50,  52,  54,  56,  57,  60,  62,  64,
            66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,
            88,  90,  92,  94,  96,  98, 100]
    
    AttackFiles = listdir(path + 'Results/Attack/GraphData/alpha/')
    
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Attack/GraphData/alpha/' + AttackFiles[0])
    for i in range(len(Missions)):
        # Attack missions will always be format of "C||P-MissionName-Sensor.csv"
        Missions[i] = Missions[i][:-4] #Remove CSV part of name
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        Path(path + "Results/Attack/GraphData/Combined/" + Missions[i]).mkdir(parents=True, exist_ok=True)
        
        # Setting the columns to 2,4,6,..,100
        TTD_CSV = pd.DataFrame(columns=range(2,102,2))
        
        # Populating them with dummy values for rows 1,2,...,50
        for j in range(2,102,2):
            TTD_CSV[j] =  [0.0] * 50
        
        # Replacing row index with 2,4,...,100
        TTD_CSV.index = pd.RangeIndex(2, 102, 2)
        
        # During testing of 2,4,6,...,100, 58 got replaced with 57. Need to replace
        # here to stay consistent
        TTD_CSV.rename({58:57},axis=0,inplace=True)
        TTD_CSV.rename({58:57},axis=1,inplace=True)
        
        for imu in tested_values:
            for of in tested_values:
                files = listdir(path + 'Results/Attack/GraphData/alpha/imu-' + str(imu) + "_of-" + str(of))
            
                # Only consolidating the Net results, matching on mission name
                target = [match for match in files if Missions[i] in match]
                temp = pd.read_csv(path + 'Results/Attack/GraphData/alpha/imu-' + str(imu) + "_of-" + str(of) + "/" + target[0])
           
                # Assign the data to a row,column named for the LPF value,
                # Rows are the OF LPF value, Columns are the ACC LPF value
                try:
                    TTD = temp[temp['FPR']==0].iloc[0].TTD
                except:
                    # This implies FPR never reached 0
                    TTD = -1.0
                
                
                TTD_CSV.at[of, imu] = TTD
        
        TTD_CSV.to_csv(path + 'Results/Attack/GraphData/Combined/' + Missions[i] + '/TTD.csv', index=True)
        
def combine_pairwise_TTD_csv(path):

    tested_values = [2,   4,   6,   8,  10,  12,  14,  16,  18,  20,
            22,  24,  26,  28,  30,  32,  34,  36,  38,  40,  42,
            44,  46,  48,  50,  52,  54,  56,  57,  60,  62,  64,
            66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,
            88,  90,  92,  94,  96,  98, 100]
    
    AttackFiles = listdir(path + 'Results/Attack/PairwiseData/alpha/')
    
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Attack/PairwiseData/alpha/' + AttackFiles[0])
    for i in range(len(Missions)):
        # Attack missions will always be format of "C||P-MissionName-Sensor-Method.csv"
        Missions[i] = Missions[i][:-4] #Remove CSV part of name
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        Path(path + "Results/Attack/PairwiseData/Combined/" + Missions[i]).mkdir(parents=True, exist_ok=True)
        
        # Setting the columns to 2,4,6,..,100
        ACCOF_TTD_CSV = pd.DataFrame(columns=range(2,102,2))
        GPSOF_TTD_CSV = pd.DataFrame(columns=range(2,102,2))
        ACCGPS_TTD_CSV = pd.DataFrame(columns=range(2,102,2))
        GPSMAG_TTD_CSV = pd.DataFrame(columns=range(2,102,2))
        
        # Populating them with dummy values for rows 1,2,...,50
        for j in range(2,102,2):
            ACCOF_TTD_CSV[j] =  [0.0] * 50
            GPSOF_TTD_CSV[j] =  [0.0] * 50
            ACCGPS_TTD_CSV[j] =  [0.0] * 50
            GPSMAG_TTD_CSV[j] =  [0.0] * 50
        
        # Replacing row index with 2,4,...,100
        ACCOF_TTD_CSV.index = pd.RangeIndex(2, 102, 2)
        GPSOF_TTD_CSV.index = pd.RangeIndex(2, 102, 2)
        ACCGPS_TTD_CSV.index = pd.RangeIndex(2, 102, 2)
        GPSMAG_TTD_CSV.index = pd.RangeIndex(2, 102, 2)
        
        # During testing of 2,4,6,...,100, 58 got replaced with 57. Need to replace
        # here to stay consistent
        ACCOF_TTD_CSV.rename({58:57},axis=0,inplace=True)
        ACCOF_TTD_CSV.rename({58:57},axis=1,inplace=True)
        GPSOF_TTD_CSV.rename({58:57},axis=0,inplace=True)
        GPSOF_TTD_CSV.rename({58:57},axis=1,inplace=True)
        ACCGPS_TTD_CSV.rename({58:57},axis=0,inplace=True)
        ACCGPS_TTD_CSV.rename({58:57},axis=1,inplace=True)
        GPSMAG_TTD_CSV.rename({58:57},axis=0,inplace=True)
        GPSMAG_TTD_CSV.rename({58:57},axis=1,inplace=True)
        
        for imu in tested_values:
            for of in tested_values:
                files = listdir(path + 'Results/Attack/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of))
            
                # Only consolidating the Net results, matching on mission name
                target = [match for match in files if Missions[i] in match]
                temp = pd.read_csv(path + 'Results/Attack/PairwiseData/alpha/imu-' + str(imu) + "_of-" + str(of) + "/" + target[0])
           
                # Assign the data to a row,column named for the LPF value,
                # Rows are the OF LPF value, Columns are the ACC LPF value
                try:
                    GPSOF_TTD = temp[temp['GPSOF(FPR)']==0].iloc[0].loc["GPSOF(TTD)"]
                except:
                    # This implies FPR never reached 0
                    GPSOF_TTD = -1.0
                
                GPSOF_TTD_CSV.at[of, imu] = GPSOF_TTD
                
                if 'GC' in Missions[i]:
                    try:
                        GPSMAG_TTD = temp[temp['GPSMAG(FPR)']==0].iloc[0].loc["GPSMAG(TTD)"]
                    except:
                        # This implies FPR never reached 0
                        GPSMAG_TTD = -1.0
                    
                    GPSMAG_TTD_CSV.at[of, imu] = GPSMAG_TTD
                else:
                    try:
                        ACCOF_TTD = temp[temp['ACCOF(FPR)']==0].iloc[0].loc["ACCOF(TTD)"]
                    except:
                        # This implies FPR never reached 0
                        ACCOF_TTD = -1.0
                    try:
                        ACCGPS_TTD = temp[temp['ACCGPS(FPR)']==0].iloc[0].loc["ACCGPS(TTD)"]
                    except:
                        # This implies FPR never reached 0
                        ACCGPS_TTD = -1.0
                    
                    ACCOF_TTD_CSV.at[of, imu] = ACCOF_TTD
                    ACCGPS_TTD_CSV.at[of, imu] = ACCGPS_TTD
        
        GPSOF_TTD_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/GPSOF_TTD.csv', index=True)
        if 'GC' in Missions[i]:
            GPSMAG_TTD_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/GPSMAG_TTD.csv', index=True)
        else:
            ACCOF_TTD_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/ACCOF_TTD.csv', index=True)
            ACCGPS_TTD_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/ACCGPS_TTD.csv', index=True)

# Helper function to present plots of results    
def graph_conf(ts, sig1, sig1_bound, sig2, sig2_bound, names=["sig1", "sig2"]):
    fig = plt.figure()
    x = ts
    
    # First signal
    plt.plot(x, sig1, label=names[0])
    # Sig1 Error 1
    plt.scatter(x, sig1 + abs(sig1_bound), marker='.', label='+'+names[0])
    # Sig1 Error 2
    plt.scatter(x, sig1 - abs(sig1_bound), marker='.', label='-'+names[0])
    
    # Second signal
    plt.plot(x, sig2, label=names[1])
    # Sig2 Error 1
    plt.scatter(x, sig2 + abs(sig2_bound), marker='.', label='+'+names[1])
    plt.scatter(x, sig2 - abs(sig2_bound), marker='.', label='-'+names[1])
    
    plt.legend(loc='lower right')
    
def graph_dis_signals(df, axlab=['Time (s)', "Disagreement (m/s)"], roll=0, timing=[]):
    fig = plt.figure()
    x = df.iloc[:,0]
    sig1 = df.iloc[:,1]
    sig2 = df.iloc[:,2]
    names = [df.iloc[:,1].name, df.iloc[:,2].name]
    
    plt.plot(x, sig1, label=names[0], color='blue')
    plt.plot(x, sig2, label=names[1], color='red')
    
    plt.xlabel(axlab[0])
    plt.ylabel(axlab[1])
    
    plt.legend()
    
    plt.fill_between(x, sig1, sig2, where=sig2>sig1, facecolor='red', alpha=0.2, interpolate=True)
    plt.fill_between(x, sig1, sig2, where=sig2<=sig1, facecolor='blue', alpha=0.2, interpolate=True)
    plt.plot(x, [0]*len(x), color="black")
    
    if roll != 0:
        plt.plot(df.TimeUS, df.Off.rolling(roll, min_periods=1).sum())
    if len(timing) == 3:
        plt.axvline(timing[1], color='red')

# Just a wrapper to cast sympy atan2 to a float
def arctan2(y, x):
    return(float(atan2(y, x)))


# Helper to cast radians to degrees
def ToDeg(x):
    return(float(x * 180 / pi))

def heading(x):
    deg = ToDeg(np.arccos(np.dot(x/norm(x), np.array([1, 0],dtype=float))))
    if(x[1] < 0):
        return 360 - deg
    else:
        return deg

# Calculates angle given that North is 0 degrees and we rotate clock-wise
def heading_series(x):
    deg = []
    for index, row in x.iterrows():
        deg.append(heading(row))
    return deg

# normalize a set of values
def norm(x):
    temp = 0
    for val in x:
        temp += (val * val)
    return(float(sqrt(temp)))


# Expecting pandas dataframe in the format of
# "Timestamp,Sensor1,Err1,Sensor2,Err2"
# Performs a confirmation of errors given sensor readings with error magnitude
# Wrap is used if the error bounds can wrap around like in a unit circle
def confirm(df, wrap=False):
    if(df.isnull().values.any()):
        print("nan found in Confirmation:")
        print(df.columns)
        exit
    if not wrap:
        subset = df.copy(deep=True)
        subset = subset.where(subset.iloc[:, 1] <= subset.iloc[:, 3], -subset)
        subset.iloc[:, 2] = subset.iloc[:, 2].abs()
        subset.iloc[:, 4] = subset.iloc[:, 4].abs()
        errors = df[((subset.iloc[:, 1] + subset.iloc[:, 2]) -
                    (subset.iloc[:, 3] - subset.iloc[:, 4])) < 0]
        if errors.empty:
            return errors
        errors.loc[:, 'Off'] = ((subset.iloc[:, 1] + subset.iloc[:, 2]) -
                                (subset.iloc[:, 3] - subset.iloc[:, 4])).abs()
    else:
        errors = pd.DataFrame(columns=df.columns.tolist() + ["Off"])
        for index, row in df.iterrows():
            Err = np.array([0, 0],dtype=float)
            # Erroneous positions closest together
            if row.iloc[1] < row.iloc[3]:
                lower = row.iloc[1] + row.iloc[2]
                upper = row.iloc[3] - row.iloc[4]
            else:
                lower = row.iloc[3] + row.iloc[4]
                upper = row.iloc[1] - row.iloc[2]
            # The idea here is if wrapping occurs, the error stretches pass
            # even the other position and ends up wrapping back around
            if (lower >= 360) or (upper <= 0):
                # Either wrapping means they confirm
                continue
            elif lower - upper >= 0:
                # Confirmed
                continue
            Err[0] = lower - upper
            # Need to consider positions in other direction, i.e., overlapping
            # wrapping. Erroneous positions furthest from one another
            if row.iloc[1] >= row.iloc[3]:
                upper = row.iloc[1] + row.iloc[2]
                lower = row.iloc[3] - row.iloc[4]
            else:
                upper = row.iloc[3] + row.iloc[4]
                lower = row.iloc[1] - row.iloc[2]
            if (lower <= 0) and (upper >= 360):
                # Both wrapping is an easy catch for confirming
                continue
            elif (lower <= 0):
                lower %= 360
                if upper - lower >= 0:
                    # confirmed
                    continue
            elif (upper >= 360):
                upper %= 360
                if upper - lower >= 0:
                    # confirmed
                    continue
            # Reaching this point with no wrapping means failed to confirm
            Err[1] = lower - upper
            errors = errors.append({errors.columns[0]: row.iloc[0],
                                    errors.columns[1]: row.iloc[1],
                                    errors.columns[2]: row.iloc[2],
                                    errors.columns[3]: row.iloc[3],
                                    errors.columns[4]: row.iloc[4],
                                    errors.columns[5]: min(abs(Err))
                              }, ignore_index=True
                            )        
    return errors

# Expecting pandas dataframe in the format of
# "Timestamp,Sensor1,Sensor2"
# Calculates the difference in the signals and the cumulative difference at
#  each step
# Wrap is used for heading where the values are modulo 360
def boundary(df, wrap=False):
    if(df.isnull().values.any()):
        print("nan found in Confirmation:")
        print(df.columns)
        exit
    if not wrap:
        subset = df.copy(deep=True)
        subset.loc[:, 'Off'] = subset.iloc[:, 1] - subset.iloc[:, 2]
        subset.loc[:, 'Disagreement'] = subset.Off.cumsum()
            
    else:
        subset = df.copy(deep=True)
        subset.loc[:, 'Off'] = 0
        for index, row in df.iterrows():
            # If difference is less than 180, just use the difference
            if abs(row.iloc[1] - row.iloc[2]) < 180:
                subset.loc[index, 'Off'] = row.iloc[1] - row.iloc[2]
            # Positive Result because row.iloc[1] < row.iloc[2] means more cw
            elif row.iloc[1] < row.iloc[2]:
                subset.loc[index, 'Off'] = 360 - (row.iloc[2] - row.iloc[1])
            # Negative result because row.iloc[1] > row.iloc[2] means more ccw
            elif row.iloc[2] <= row.iloc[1]:
                subset.loc[index, 'Off'] = row.iloc[1] - row.iloc[2] - 360
        subset.loc[:, 'Disagreement'] = subset.Off.cumsum()
    return subset

# Helper function that expects the results of the Boundary function above
#  Calculates the max disagreement for the benign and attack portion of a
#  mission given a window size. Used for determining viable window sizes
def peak_compare(sig1, timing, roll):
    benign = sig1[(sig1['TimeUS'] < timing[1]) & (sig1['TimeUS'] > timing[0])]
    attack = sig1[(sig1['TimeUS'] < timing[2]) & (sig1['TimeUS'] > timing[1])]
    benign_windowed = benign.Off.rolling(roll, min_periods=1)
    benign_peak = max(benign_windowed.sum().max(), abs(benign_windowed.sum().min()))
    attack_windowed = attack.Off.rolling(roll, min_periods=1)
    attack_peak = max(attack_windowed.sum().max(), abs(attack_windowed.sum().min()))
    return(attack_peak/benign_peak)

def yaw_from_mag_tc(magx, magy, magz, pitch, roll):
    mag_norm = sqrt((magx*magx) + (magy * magy) + (magz + magz))
    magx = magx/mag_norm
    magy = magy/mag_norm
    magz = magz/mag_norm
    return(float(atan2((-magy*cos(roll) + magz*sin(roll)),
                       (magx*cos(pitch) + magy*sin(pitch)*sin(roll) +
                        magz*sin(pitch)*cos(roll)))))


def yaw_from_mag_hw(magx, magy, dec):
    if magy > 0:
        return(float(90 - ToDeg(arctan2(magx, magy))))
    elif magy < 0:
        return(float(270 - ToDeg(arctan2(magx, magy))))
    elif magy == 0 and magx < 0:
        return(180)
    else:
        return(0)


# Reference for below equations:
# https://www.nxp.com/docs/en/application-note/AN4248.pdf
# Provide a dataframe in the format of
# Acc F, Acc R, Acc D, GyR, GyP, GyY
def attitude(df):
    pitch = lambda f, r, d : arctan2(f / sqrt(r*r + d*d))
    roll = lambda f, r, d : arctan2(r / sqrt(f*f + d*d))
    yaw = lambda x, y : arctan2()
    

# Yaw is Psi, Pitch is Theta, Roll is Phi
def rot_mat_bf_to_ef(roll, pitch, yaw):
    cy = float(cos(yaw))
    cp = float(cos(pitch))
    cr = float(cos(roll))
    sy = float(sin(yaw))
    sp = float(sin(pitch))
    sr = float(sin(roll))
    ret = np.array([[cy * cp, cy * sr * sp - cr * sy, sr * sy + cr * cy * sp],
                    [cp * sy, cr * cy + sr * sy * sp, cr * sy * sp - cy * sr],
                    [    -sp,                cp * sr,                cr * cp]])
    return(ret)


# df: Dataframe containing the column names below as well as rotation matrix
#     with the matrix elements stored in columns as m00, m01, m02
#                                                   m10, m11, m12
#                                                   m20, m21, m22
# inNames: column names holding input inforation in Front,Right,Down frame
# outNames: column names where North, East, and Down data will be stored
# Rotates from body frame to intertial frame
def bf_to_ef(df, inNames, outNames):
    for index, row in df.iterrows():
        bf = pd.DataFrame([row.loc[inNames[0]], row.loc[inNames[1]],
                           row.loc[inNames[2]]])
        # Matrix is in IF to BF format, transpose to get the inverse
        rot = pd.DataFrame([[row.loc['m00'], row.loc['m10'], row.loc['m20']],
                            [row.loc['m01'], row.loc['m11'], row.loc['m21']],
                            [row.loc['m02'], row.loc['m12'], row.loc['m22']]])
        ef = rot.dot(bf)
        df.iloc[index, df.columns.get_loc(outNames[0])] = ef.iloc[0]
        df.iloc[index, df.columns.get_loc(outNames[1])] = ef.iloc[1]
        df.iloc[index, df.columns.get_loc(outNames[2])] = ef.iloc[2]
    return(ef)


# df: Dataframe containing aF, aR, and aD as well as rotation matrix
#     with the matrix elements stored in columns as m00, m01, m02
#                                                   m10, m11, m12
#                                                   m20, m21, m22
# averaging the first rows measurements helps remove noise from sensors but
# due to the multicopter attitude changing during flight the averaged noise
# also needs to be rotated to stay in correct frame.
def reduce_noise(df, rows):
    avgNoise = pd.DataFrame([df['aF'][:rows].mean(), df['aR'][:rows].mean(),
                             df['aD'][:rows].mean()])
    for index, row in df.iterrows():
        # Matrix is in IF to BF format, transpose to get the inverse
        rot = pd.DataFrame([[row.loc['m00'], row.loc['m10'], row.loc['m20']],
                            [row.loc['m01'], row.loc['m11'], row.loc['m21']],
                            [row.loc['m02'], row.loc['m12'], row.loc['m22']]])
        ef = rot.dot(avgNoise)
        df.iloc[index, df.columns.get_loc('aF')] -= ef.iloc[0]
        df.iloc[index, df.columns.get_loc('aR')] -= ef.iloc[1]
        df.iloc[index, df.columns.get_loc('aD')] -= ef.iloc[2]
    return(ef)


def main():
    combine_pairwise_TTD_csv('./Data/2022-05-01/')
    combine_system_TTD_csv('./Data/2022-05-01/')
    combine_FPR_TPR_csv('./Data/2022-05-01/')
    
if __name__ == "__main__":
    main()
