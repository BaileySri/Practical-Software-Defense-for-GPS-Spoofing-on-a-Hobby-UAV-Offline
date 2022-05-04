# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 02:41:03 2021

Preface:
    All angles such as roll, pitch ,or yaw are assumed to be in radians
@author: Bailey K. Srimoungchanh
"""
import numpy as np
import pandas as pd
from os import listdir, mkdir
from sympy import sin, cos, pi, atan2, sqrt

# Helper function to put together the CSVs generated for LPF testing
def combine_csv(path):
    BenignFiles = listdir(path + 'Results/Benign/PairwiseData/alpha/')
    BenignFiles.sort(key=int)
    AttackFiles = listdir(path + 'Results/Attack/PairwiseData/alpha/')
    AttackFiles.sort(key=int)
    
    #Processing the Benign files first
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Benign/PairwiseData/alpha/' + BenignFiles[0])
    for i in range(len(Missions)):
        Missions[i] = Missions[i].split('-')[1]
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        try:
            mkdir(path + 'Results/Benign/PairwiseData/Combined/')
        except:
            pass
        try:
            mkdir(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i])
        except:
            pass
        
        GPSOF_CSV = pd.DataFrame()
        IMUOF_CSV = pd.DataFrame()
        IMUGPS_CSV = pd.DataFrame()
        
        for j in BenignFiles:
            files = listdir(path + 'Results/Benign/PairwiseData/alpha/' + j)
            
            # Only consolidating the Net results, matching on mission name and net
            target = [match for match in files if Missions[i] in match and 'Net' in match]
            temp = pd.read_csv(path + 'Results/Benign/PairwiseData/alpha/' + j + '/' + target[0])
            
            # Assign the data to a column named for the LPF value
            GPSOF_CSV[j] = temp['GPSOF(FPR)']
            IMUOF_CSV[j] = temp['ACCOF(FPR)']
            IMUGPS_CSV[j] = temp['ACCGPS(FPR)']
        
        GPSOF_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/GPSOF.csv',index=False)
        IMUOF_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/IMUOF.csv',index=False)
        IMUGPS_CSV.to_csv(path + 'Results/Benign/PairwiseData/Combined/' + Missions[i] + '/IMUGPS.csv',index=False)
    
    #Processing the Attack files second
    # Collecting a list of mission types in the folder
    Missions = listdir(path + 'Results/Attack/PairwiseData/alpha/' + AttackFiles[0])
    for i in range(len(Missions)):
        temp = Missions[i].split('-')
        Missions[i] = temp[1] + "-" + temp[2]
    Missions = np.unique(Missions)
    
    for i in range(len(Missions)):
        # Make combined directory
        try:
            mkdir(path + 'Results/Attack/PairwiseData/Combined/')
        except:
            pass
        try:
            mkdir(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i])
        except:
            pass
        
        GPSOF_CSV = pd.DataFrame()
        IMUOF_CSV = pd.DataFrame()
        IMUGPS_CSV = pd.DataFrame()
        
        for j in BenignFiles:
            files = listdir(path + 'Results/Attack/PairwiseData/alpha/' + j)
            
            # Only consolidating the Net results, matching on mission name and net
            target = [match for match in files if Missions[i] in match and 'Net' in match]
            temp = pd.read_csv(path + 'Results/Attack/PairwiseData/alpha/' + j + '/' + target[0])
            
            # Assign the data to a column named for the LPF value
            GPSOF_CSV[j] = temp['GPSOF(FPR)'].append(temp['GPSOF(TPR)'])
            IMUOF_CSV[j] = temp['ACCOF(FPR)'].append(temp['ACCOF(TPR)'])
            IMUGPS_CSV[j] = temp['ACCGPS(FPR)'].append(temp['ACCGPS(TPR)'])
        
        GPSOF_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/GPSOF.csv',index=False)
        IMUOF_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUOF.csv',index=False)
        IMUGPS_CSV.to_csv(path + 'Results/Attack/PairwiseData/Combined/' + Missions[i] + '/IMUGPS.csv',index=False)
    
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
    combine_csv('./Data/2022-05-01/')
    
if __name__ == "__main__":
    main()
