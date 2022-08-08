# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:38:16 2021

@author: Bailey K. Srimoungchanh
"""
from Confirmation import boundary, bf_to_ef, norm, reduce_noise
from Confirmation import ToDeg, heading, peak_compare
from functools import reduce
from math import sqrt, cos, tan
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat
from time import time    
    
def process(date, missions, times, live=False, imulpf=0, oflpf=0):
    base = "./Data/" + date
    Path(base + "/Results/Attack/GraphData/").mkdir(parents=True, exist_ok=True)
    Path(base + "/Results/Benign/GraphData/").mkdir(parents=True, exist_ok=True)
    Path(base + "/Results/Attack/PairwiseData/").mkdir(parents=True, exist_ok=True)
    Path(base + "/Results/Benign/PairwiseData/").mkdir(parents=True, exist_ok=True)

    test_thresholds = 30
    tests = zip(missions, times)
    for name, timing in tests:
        
        if(len(timing) == 3):
            dir_type = "Attack/"
        else:
            dir_type = "Benign/"
        
        if imulpf != 0 or oflpf != 0:
            graphDataDir = base + "/Results/" + dir_type + "GraphData/alpha/imu-" + str(int(imulpf*100)) + '_of-' + str(int(oflpf * 100)) + "/"
            Path(graphDataDir).mkdir(parents=True, exist_ok=True)
            graphData = base + "/Results/" + dir_type + "GraphData/alpha/imu-" + str(int(imulpf*100)) + '_of-' + str(int(oflpf * 100)) + "/" + name[:-3] + "csv"
        else:
            graphData = base + "/Results/" + dir_type + "GraphData/" + name[:-3] + "csv"

        pairwiseData = base + "/Results/" + dir_type + "PairwiseData/" + name[:-4]
        files = []
        
        CNF_Range = 5 if live else 4
        for i in range(1,CNF_Range):
            files.append(base + "/" + date + "-CNF" + str(i) + "-" + name)
        for i in range(1,3):
            files.append(base + "/" + date + "-ACO" + str(i) + "-" + name)
        CNF1 = pd.read_csv(files[0])
        CNF2 = pd.read_csv(files[1])
        CNF3 = pd.read_csv(files[2])
        if live:
            CNF4 = pd.read_csv(files[3])
            ACO1 = pd.read_csv(files[4])
            ACO2 = pd.read_csv(files[5])
        else:
            ACO1 = pd.read_csv(files[3])
            ACO2 = pd.read_csv(files[4])
        if live:
            CNFs = [CNF1, CNF2, CNF3, CNF4]
        else:
            CNFs = [CNF1, CNF2, CNF3]
        ACOs = [ACO1, ACO2]
        CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
        ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)     
        CNF = CNF.drop(CNF[CNF.TimeUS < timing[0]].index)
        CNF = CNF.drop(CNF[CNF.TimeUS > timing[len(timing) - 1]].index).reset_index(drop=True)
        ACO = ACO.drop(ACO[ACO.TimeUS < timing[0]].index)
        ACO = ACO.drop(ACO[ACO.TimeUS > timing[len(timing) - 1]].index).reset_index(drop=True)
        
        if(not live):
            if not 'gpSA' in CNF.columns:
                CNF['gpSA'] = 0.60 #60cm/s is what we expect for gpSA in live flight
        coverages = {"3Axis":{"ACCOF":{},"ACCGPS":{},"GPSOF":{}},
                     "Net"   :{"ACCOF":{},"ACCGPS":{},"GPSOF":{}},
                     "GC"    :{"GPSMAG":{}, "GPSOF":{}}}
        
        if((imulpf != 0) or (oflpf != 0)):
            # LPF directory
            Path(base + "/Results/" + dir_type + 
                  "PairwiseData/alpha/imu-" + str(int(imulpf*100)) +
                  '_of-' + str(int(oflpf * 100)) + "/").mkdir(parents=True, exist_ok=True)
        
        # Low Pass Filter on OF results
        # LPF values are the % of the new data being used
        if(oflpf != 0):
            for index, row in ACO.iterrows():
                if(index == 0):
                    continue
                ACO.loc[index, "COFN"] = ACO.loc[index, "COFN"]*(oflpf) + ACO.loc[index-1, "COFN"]*(1-oflpf)
                ACO.loc[index, "COFE"] = ACO.loc[index, "COFE"]*(oflpf) + ACO.loc[index-1, "COFE"]*(1-oflpf)
                ACO.loc[index, "POFN"] = ACO.loc[index-1, "COFN"]
                ACO.loc[index, "POFE"] = ACO.loc[index-1, "COFE"]
                
            for index, row in CNF.iterrows():
                if(index == 0):
                    continue
                CNF.loc[index, "COFN"] = CNF.loc[index, "COFN"]*(oflpf) + CNF.loc[index-1, "COFN"]*(1-oflpf)
                CNF.loc[index, "COFE"] = CNF.loc[index, "COFE"]*(oflpf) + CNF.loc[index-1, "COFE"]*(1-oflpf)
                CNF.loc[index, "POFN"] = CNF.loc[index-1, "COFN"]
                CNF.loc[index, "POFE"] = CNF.loc[index-1, "COFE"]
                
        # Low Pass Filter on IMU results
        if(imulpf != 0):
            for index, row in ACO.iterrows():
                if(index == 0):
                    continue
                ACO.loc[index, "CAN"] = ACO.loc[index, "CAN"]*(imulpf) + ACO.loc[index-1, "CAN"]*(1-imulpf)
                ACO.loc[index, "CAE"] = ACO.loc[index, "CAE"]*(imulpf) + ACO.loc[index-1, "CAE"]*(1-imulpf)
                ACO.loc[index, "CAD"] = ACO.loc[index, "CAD"]*(imulpf) + ACO.loc[index-1, "CAD"]*(1-imulpf)
                
                
            for index, row in CNF.iterrows():
                if(index == 0):
                    continue
                CNF.loc[index, "CAN"] = CNF.loc[index, "CAN"]*(imulpf) + CNF.loc[index-1, "CAN"]*(1-imulpf)
                CNF.loc[index, "CAE"] = CNF.loc[index, "CAE"]*(imulpf) + CNF.loc[index-1, "CAE"]*(1-imulpf)
                CNF.loc[index, "CAD"] = CNF.loc[index, "CAD"]*(imulpf) + CNF.loc[index-1, "CAD"]*(1-imulpf)
                

#---IMU and OF---#
    #Velocity Change
    #3D
        North = pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],'OF':ACO['COFN']-ACO['POFN'], 'Acc':ACO['CAN']})
        East = pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],'OF':ACO['COFE']-ACO['POFE'], 'Acc':ACO['CAE']})
        ACO_IMUOF_N = boundary(North)
        ACO_IMUOF_E = boundary(East)

        
    #Scalar
        North = ACO['COFN'] - ACO['POFN']
        East = ACO['COFE'] - ACO['POFE']
        # Added IR below to make it easier to view variables when debugging
        IR = pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],
                                   'OF':(pd.DataFrame(data = {"N":North, "E":East})).apply(norm, axis=1),
                                   'ACC':(ACO[['CAN','CAE']]).apply(norm,axis=1)})
        ACO_IMUOF_Scalar = boundary(IR)
              

#---GPS and Magnetometer---#
        #Mag GC
        MagGC = map(ToDeg, map(np.arctan2,CNF['m10'].values,CNF['m00'].values))
        MagGC = [x + 360 if x < 0 else x for x in MagGC]
        
        #GPS GC
        dot = CNF[['CGpN']]
        det = -CNF[['CGpE']]
        GpsGC = map(ToDeg,list(map(np.arctan2,det.values,dot.values)))
        GpsGC = [360 - x if x > 0 else abs(x) for x in GpsGC]

        # Added IR below to make it easier to view variables when debugging
        IR = pd.DataFrame( data = {'TimeUS':CNF['TimeUS'],'MagGC':MagGC, 'GPSGC':GpsGC})
        CNF_GPSMAG_GC = boundary(IR, wrap=True)


#---Accelerometer and GPS---#
    #3D
        North = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpN']-CNF['PGpN'],'Acc':CNF['CAN']})
        East = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpE']-CNF['PGpE'],'Acc':CNF['CAE'],})
        Down = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpD']-CNF['PGpD'],'Acc':CNF['CAD']})
        CNF_IMUGPS_N = boundary(North)
        CNF_IMUGPS_E = boundary(East)
        CNF_IMUGPS_D = boundary(Down)

                
    #Scalar
        North = CNF['CGpN'] - CNF['PGpN']
        East = CNF['CGpE'] - CNF['PGpE']
        Down = CNF['CGpD'] - CNF['PGpD']
        IR = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],
                                   'GPS':(pd.DataFrame(data ={'N':North, 'E':East, 'D':Down})).apply(norm, axis=1),
                                   'ACC':(CNF[['CAN','CAE','CAD']]).apply(norm,axis=1)})
        CNF_IMUGPS_Scalar = boundary(IR)

#---GPS and OF---#
    #3D
        North = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpN'],'OF':CNF['COFN']})
        East = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpE'],'OF':CNF['COFE']})
        CNF_GPSOF_N = boundary(North)
        CNF_GPSOF_E = boundary(East)

    #Scalar
        IR = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],
                                           'OF':(CNF[['COFN','COFE']]).apply(norm, axis=1),
                                           'GPS':(CNF[['CGpN','CGpE']]).apply(norm,axis=1)})
        CNF_GPSOF_Scalar = boundary(IR)
        
    #Ground Course
        dot = CNF[['COFN']]
        det = -CNF[['COFE']]
        OFGC = map(ToDeg, map(np.arctan2, det.values, dot.values))
        OFGC = [360 - x if x > 0 else abs(x) for x in OFGC]
        
        IR = pd.DataFrame( data = {'TimeUS':CNF['TimeUS'],
                                            'OFGC':OFGC,
                                            'GPSGC':GpsGC})
        CNF_GPSOF_GC = boundary(IR, wrap=True)
        if len(timing) > 2:
            xaxis = np.arange(1, 31, 1)
            yaxis = [peak_compare(ACO_IMUOF_N, timing, x) for x in xaxis]
        

def main():
    # Simulation Data
    # date = "2022-04-19"
    # missions = [
    #             "C-Adversarial-GPS.txt",
    #             "C-Adversarial-OF.txt",
    #             "C-Delivery.txt",
    #             "C-Idle-GPS.txt",
    #             "C-Idle-OF.txt",
    #             "P-Adversarial-GPS.txt",
    #             "P-Delivery.txt"
    #     ]
    # times = [
    #             [62315897,87260915,147860832],
    #             [62315897,87468332,128261175],
    #             [62308400,136023902],
    #             [51028747,73264016,133858935],
    #             [51028747,73261517,113858605],
    #             [50900465,61640334,82040504],
    #             [50400665,86400426]
    #         ]
    
    date = "2022-05-01"
    missions = [
                "C-Delivery.txt",
                "C-Idle-GPS.txt",
                "C-Idle-OF.txt"
        ]
    times = [
                [121260000,193595505],
                [264334818,274334818,291660048],
                [286066318,296066318,310068663]
        ]
    
    # process(date, missions, times, live=True)
    oflpf_args = np.linspace(0, 1, 51)
    imulpf_args = np.linspace(0, 1, 51)
    combined = [(a, b) for a in oflpf_args for b in imulpf_args]
    oflpf_args, imulpf_args = zip(*combined)
    
    start = time()
    # I recommend anyone running this script to adjust the below Pool
    # parameter for their system. Otherwise you may get a pagefile too small
    # error
    # with Pool(8) as pool:
    #     pool.starmap(process, zip(repeat(date), repeat(missions), repeat(times),
    #                               repeat(True), imulpf_args, oflpf_args))
    process(date, missions, times, True, 1.0, 0.2)
    end = time()
    
    print("Took " + str(int((end-start))) + " (s)")


    
if __name__ == "__main__":
    main()

#------------------Early Development Functions-------------------------------#
# All functions present below this line were for data analysis during the
# early part of developing the ArduPilot Sensor Confirmation implementation.
# Some of the early functions and data files may not work on current
# processing due to change in logging structure. They are mainly kept here
# as a logging of prior work done.

# Any function prefixed with data_ was used to generate the relevant dataframe
# for analysis.

# Attack Copter ACO data for confirming Optical Flow and Acc, Mission starts at 148s,
# ends at 270s. Attack begins at 176.6s and ends at 232s
def data_8_16_ACO():
    ACO1 = pd.read_csv("./Data/2021-08-16/2021-08-16-ACO1.txt")
    ACO2 = pd.read_csv("./Data/2021-08-16/2021-08-16-ACO2.txt")
    ACOs = [ACO1, ACO2]
    ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)
    # Whole flight is relevant
    return(ACO)

# Attack Copter CNF data for comparison to SNS data, Mission starts at 210s,
# ends at 428s
def data_8_16_CNF():
    CNF1 = pd.read_csv("./Data/2021-08-16/2021-08-16-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-08-16/2021-08-16-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-08-16/2021-08-16-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Attack Copter SNS data for comparison to CNF data. 210s to 428s
def data_8_16_SNS():
    SNS1 = pd.read_csv("./Data/2021-08-16/2021-08-16-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-08-16/2021-08-16-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-08-16/2021-08-16-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-08-16/2021-08-16-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Attack Copter ACO data for confirming Optical Flow and Acc, Mission starts at 210s,
# ends at 401s. Attack begins at 215.8s and ends at 346.4s
def data_8_11_ACO():
    ACO1 = pd.read_csv("./Data/2021-08-11/2021-08-11-ACO1.txt")
    ACO2 = pd.read_csv("./Data/2021-08-11/2021-08-11-ACO2.txt")
    ACOs = [ACO1, ACO2]
    ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)
    # Whole flight is relevant
    return(ACO)

# Attack Copter CNF data for comparison to SNS data, Mission starts at 210s,
# ends at 428s
def data_8_11_CNF():
    CNF1 = pd.read_csv("./Data/2021-08-11/2021-08-11-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-08-11/2021-08-11-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-08-11/2021-08-11-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Attack Copter SNS data for comparison to CNF data. 210s to 428s
def data_8_11_SNS():
    SNS1 = pd.read_csv("./Data/2021-08-11/2021-08-11-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-08-11/2021-08-11-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-08-11/2021-08-11-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-08-11/2021-08-11-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Square Copter ACO data for confirming Optical Flow and Acc, Mission starts at 229s,
# ends at 401s
def data_8_10_ACO():
    ACO1 = pd.read_csv("./Data/2021-08-10/2021-08-10-ACO1.txt")
    ACO2 = pd.read_csv("./Data/2021-08-10/2021-08-10-ACO2.txt")
    ACOs = [ACO1, ACO2]
    ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)
    # Whole flight is relevant
    return(ACO)

# Square Copter CNF data for comparison to SNS data, Mission starts at 200s,
# ends at 378s
def data_8_10_CNF():
    CNF1 = pd.read_csv("./Data/2021-08-10/2021-08-10-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-08-10/2021-08-10-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-08-10/2021-08-10-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Square Copter SNS data for comparison to CNF data. 200s to 378s
def data_8_10_SNS():
    SNS1 = pd.read_csv("./Data/2021-08-10/2021-08-10-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-08-10/2021-08-10-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-08-10/2021-08-10-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-08-10/2021-08-10-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Square Copter ACO data for confirming Optical Flow and Acc, Mission starts at 237s,
# ends at 422s
def data_8_09_ACO():
    ACO1 = pd.read_csv("./Data/2021-08-09/2021-08-09-ACO1.txt")
    ACO2 = pd.read_csv("./Data/2021-08-09/2021-08-09-ACO2.txt")
    ACOs = [ACO1, ACO2]
    ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)
    # Whole flight is relevant
    return(ACO)

# Square Copter CNF data for comparison to SNS data, Mission starts at 237s,
# ends at 422s
def data_8_09_CNF():
    CNF1 = pd.read_csv("./Data/2021-08-09/2021-08-09-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-08-09/2021-08-09-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-08-09/2021-08-09-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Square Copter SNS data for comparison to CNF data
def data_8_09_SNS():
    SNS1 = pd.read_csv("./Data/2021-08-09/2021-08-09-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-08-09/2021-08-09-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-08-09/2021-08-09-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-08-09/2021-08-09-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Square Copter CNF data for comparison to SNS data, Mission starts at 173s,
# ends at 353s
def data_8_03_CNF():
    CNF1 = pd.read_csv("./Data/2021-08-03/2021-08-03-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-08-03/2021-08-03-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-08-03/2021-08-03-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Square Copter SNS data for comparison to CNF data
def data_8_03_SNS():
    SNS1 = pd.read_csv("./Data/2021-08-03/2021-08-03-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-08-03/2021-08-03-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-08-03/2021-08-03-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-08-03/2021-08-03-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Square Copter CNF data for comparison to SNS data, Mission starts at 157s,
# ends at 341s
def data_7_26_CNF():
    CNF1 = pd.read_csv("./Data/2021-07-26/2021-07-26-CNF1.txt")
    CNF2 = pd.read_csv("./Data/2021-07-26/2021-07-26-CNF2.txt")
    CNF3 = pd.read_csv("./Data/2021-07-26/2021-07-26-CNF3.txt")
    CNFs = [CNF1, CNF2, CNF3]
    CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
    # Whole flight is relevant
    return(CNF)

# Square Copter SNS data for comparison to CNF data
def data_7_26_SNS():
    SNS1 = pd.read_csv("./Data/2021-07-26/2021-07-26-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-07-26/2021-07-26-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-07-26/2021-07-26-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-07-26/2021-07-26-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Copter flying the AuoAttack plan with M8 GPS and Optical Flow, Attack from
# 126 seconds to 212
def data_7_07_Copter():
    SNS1 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS1-C-Atk-OF.txt")
    SNS2 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS2-C-Atk-OF.txt")
    SNS3 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS3-C-Atk-OF.txt")
    SNS4 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS4-C-Atk-OF.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Plane flying the AuoAttack plan with M8 GPS and Optical Flow, Attack from
# 123.47 seconds to 160.43
def data_7_07_Plane():
    SNS1 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS1-P-Atk-OF.txt")
    SNS2 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS2-P-Atk-OF.txt")
    SNS3 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS3-P-Atk-OF.txt")
    SNS4 = pd.read_csv("./Data/2021-07-07/2021-07-07-SNS4-P-Atk-OF.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)

# Copter flying the benign square path with M8 GPS and Optical Flow
def data_7_03_Square_Plane():
    SNS1 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS1-S-P-OF.txt")
    SNS2 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS2-S-P-OF.txt")
    SNS3 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS3-S-P-OF.txt")
    SNS4 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS4-S-P-OF.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Copter flying the benign square path with M8 GPS and Optical Flow
def data_7_03_Square_Copter():
    SNS1 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS1-S-C-OF.txt")
    SNS2 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS2-S-C-OF.txt")
    SNS3 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS3-S-C-OF.txt")
    SNS4 = pd.read_csv("./Data/2021-07-03/2021-07-03-SNS4-S-C-OF.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Plane Square flight with default sensor values
def data_6_30_Square():
    SNS1 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS1-Square.txt")
    SNS2 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS2-Square.txt")
    SNS3 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS3-Square.txt")
    SNS4 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS4-Square.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Plane Circle flight with default sensor values
def data_6_30_Circle():
    SNS1 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS1-Circle.txt")
    SNS2 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS2-Circle.txt")
    SNS3 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS3-Circle.txt")
    SNS4 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS4-Circle.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Plane AutoAttack with 10m offset, ZEDF9P, Attack starts at 87.6 seconds
# and ends at 147.7 seconds
def data_6_30_Attack():
    SNS1 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-30/2021-06-30-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# AutoAttack with 1cm offset, ZEDF9P
def data_6_23():
    SNS1 = pd.read_csv("./Data/2021-06-23/2021-06-23-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-23/2021-06-23-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-23/2021-06-23-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-23/2021-06-23-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# AutoAttack with 1cm offset, default sensors
def data_6_20():
    SNS1 = pd.read_csv("./Data/2021-06-20/2021-06-20-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-20/2021-06-20-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-20/2021-06-20-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-20/2021-06-20-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# AutoAttack with ZEDF9P GPS Positional Accuracy
def data_6_16_ZED_ATK():
    SNS1 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS1-ZED-ATK.txt")
    SNS2 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS2-ZED-ATK.txt")
    SNS3 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS3-ZED-ATK.txt")
    SNS4 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS4-ZED-ATK.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation of a light yaw attack with a 1cm offset, sensors are default
def data_6_16_1cm():
    SNS1 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS1-1cm.txt")
    SNS2 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS2-1cm.txt")
    SNS3 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS3-1cm.txt")
    SNS4 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS4-1cm.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation of Turnback Attack with an overshoot of 2m and default sensors
def data_6_16():
    SNS1 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-16/2021-06-16-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation with AutoAttack testing pushing drone west, sensors are default
def data_6_13_ZEDF9P():
    SNS1 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS1-ZEDF9P.txt")
    SNS2 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS2-ZEDF9P.txt")
    SNS3 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS3-ZEDF9P.txt")
    SNS4 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS4-ZEDF9P.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation with AutoBenign, GPS is NEOM8N
def data_6_13_NEOM8N():
    SNS1 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS1-NEOM8N.txt")
    SNS2 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS2-NEOM8N.txt")
    SNS3 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS3-NEOM8N.txt")
    SNS4 = pd.read_csv("./Data/2021-06-13/2021-06-13-SNS4-NEOM8N.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation with AutoAttack testing pushing drone west, sensors are default
def data_6_08():
    SNS1 = pd.read_csv("./Data/2021-06-08/2021-06-08-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-08/2021-06-08-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-08/2021-06-08-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-08/2021-06-08-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Simulation with ArduCopter default noise settings on the AutoBenign test
def data_6_05_GroundTruth():
    SNS1 = pd.read_csv("./Data/2021-06-05/2021-06-05-SNS1-GroundTruth.txt")
    SNS2 = pd.read_csv("./Data/2021-06-05/2021-06-05-SNS2-GroundTruth.txt")
    SNS3 = pd.read_csv("./Data/2021-06-05/2021-06-05-SNS3-GroundTruth.txt")
    SNS4 = pd.read_csv("./Data/2021-06-05/2021-06-05-SNS4-GroundTruth.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# benign simulation with no accelerometer or gyro noise
def data_6_02_NoAccNoise():
    SNS1 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS1-NoAccNoise.txt")
    SNS2 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS2-NoAccNoise.txt")
    SNS3 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS3-NoAccNoise.txt")
    SNS4 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS4-NoAccNoise.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# benign simulation after having modified noise simulation in source code
def data_6_02():
    SNS1 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-06-01/2021-06-02-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# simulation data with ZEOF9P specs
def data_6_01_ZEOF9P():
    SNS1 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS1-ZEOF9P.txt")
    SNS2 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS2-ZEOF9P.txt")
    SNS3 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS3-ZEOF9P.txt")
    SNS4 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS4-ZEOF9P.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# first set of simulation data incorporating realistic sensor noise
def data_6_01_benign():
    SNS1 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS1-realistic-benign.txt")
    SNS2 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS2-realistic-benign.txt")
    SNS3 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS3-realistic-benign.txt")
    SNS4 = pd.read_csv("./Data/2021-06-01/2021-06-01-SNS4-realistic-benign.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    return(SNS)


# Attack of 7m with a 40m fence
def data_4_30_70():
    SNS1 = pd.read_csv("./Data/2021-04-30/2021-04-30-7_0-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-30/2021-04-30-7_0-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-30/2021-04-30-7_0-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-30/2021-04-30-7_0-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # Whole flight is relevant
    # Attack is between 2m 26.8s and 2m 34.2s
    return(SNS)


# Attack of 70cm after a few minutes of flying in circles
def data_4_30_07():
    SNS1 = pd.read_csv("./Data/2021-04-30/2021-04-30-0_7-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-30/2021-04-30-0_7-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-30/2021-04-30-0_7-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-30/2021-04-30-0_7-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # relevant portion of flight is first 6 minutes
    # Attack is between 4m 44.8s and 5m 54.8s
    return(SNS[SNS.TimeUS < 360000000].reset_index(drop=True))


# Simulated attack of 5.6m with a 40m fence, attack begins at 228s
def data_4_28_56():
    SNS1 = pd.read_csv("./Data/2021-04-28/2021-04-28-5_6-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-28/2021-04-28-5_6-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-28/2021-04-28-5_6-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-28/2021-04-28-5_6-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # relevant portion of flight is between 3 and 5 minutes
    return(SNS[SNS.TimeUS < 300000000].reset_index(drop=True))


# Simulated attack of 70cm with a 40m fence
def data_4_28_07():
    SNS1 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_7-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_7-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_7-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_7-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # relevant portion of flight is first 2.5 minutes
    return(SNS[SNS.TimeUS < 150000000].reset_index(drop=True))


# Simulated attack of 10cm with a 40m fence
def data_4_28_01():
    SNS1 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_1-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_1-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_1-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_1-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # relevant portion of flight is first minutes and 40 seconds
    return(SNS[SNS.TimeUS < 160000000].reset_index(drop=True))


# Simulated attack with no offset with a 40m fence
def data_4_28_00():
    SNS1 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_0-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_0-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_0-SNS3.txt")
    SNS4 = pd.read_csv("./Data/2021-04-28/2021-04-28-0_0-SNS4.txt")
    SNSs = [SNS1, SNS2, SNS3, SNS4]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    # relevant portion of flight is first 5 minutes
    return(SNS[SNS.TimeUS < 300000000].reset_index(drop=True))


def data_4_23():
    SNS1 = pd.read_csv("./Data/2021-04-23-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-23-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-23-SNS3.txt")
    SNSs = [SNS1, SNS2, SNS3]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    return(SNS.iloc[19929:, ].reset_index(drop=True))


def data_4_12():
    SNS1 = pd.read_csv("./Data/2021-04-12-SNS1.txt")
    SNS2 = pd.read_csv("./Data/2021-04-12-SNS2.txt")
    SNS3 = pd.read_csv("./Data/2021-04-12-SNS3.txt")
    SNSs = [SNS1, SNS2, SNS3]
    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)
    return(SNS.reset_index(drop=True))


def data_3_31():
    SNS = pd.read_csv("./Data/2021-03-31-Data-SNS1.txt")
    SNS = pd.merge(SNS, pd.read_csv("./Data/2021-03-31-Data-SNS2.txt"), on='TimeUS')
    subset1 = SNS[:21669] # Fly away
    subset2 = SNS[21669:48020]
    subset3 = SNS[48020:] # Mission
    return(subset3)
# -------------------------------------------------------------------------- #

# Moved to its own function so initialization doesn't have to be repeated in
# debugging
def acc_init(df, rows=1000):
    # Initialize columns for earth frame accelerometer
    df['aN'] = 0
    df['aE'] = 0
    df['aDv'] = 0

    # Average the accelerometer over rows # of readings. This varies by data,
    # then bundle into the offset parameter for bf_to_ef
    reduce_noise(df, rows)
    # offset = pd.DataFrame([df['aF'][:1000].mean(), df['aR'][:1000].mean(),
    #                      df['aD'][:1000].mean()])
    # Rotate body frame to earth frame accelerometer
    bf_to_ef(df, ['aF', 'aR', 'aD'], ['aN', 'aE', 'aDv'])


# Helper function to run the confirmation tests across N,E,D when confirming
# by N, E, D components instead of as vector length
def vel_confirm(df):
    ret1 = confirm(df[['TimeUS','dAccVelN','errAcc','dGpsVelN','errGps']]).drop(['dAccVelN','errAcc','dGpsVelN','errGps'],axis=1).rename(columns={'Off':'Noff'})
    ret2 = confirm(df[['TimeUS','dAccVelE','errAcc','dGpsVelE','errGps']]).drop(['dAccVelE','errAcc','dGpsVelE','errGps'],axis=1).rename(columns={'Off':'Eoff'})
    ret3 = confirm(df[['TimeUS','dAccVelD','errAcc','dGpsVelD','errGps']]).drop(['dAccVelD','errAcc','dGpsVelD','errGps'],axis=1).rename(columns={'Off':'Doff'})
    rets = [ret1, ret2, ret3]
    ret = pd.merge(ret1, ret2, how='outer', on='TimeUS')
    ret = pd.merge(ret, ret3, how='outer', on='TimeUS')
    return(ret)


def OF(df):
    # OF Updates every 55ms
    # Y axis is Pitch, X axis is Roll
    # RF Updates every 50ms
    # Going to use most recent RF measurement for OF calculation

    # Error constants
    GYR_ERR = 0.00007  # radians/s @ 10Hz
    LT5_RF_ERR = 0.01  # meters
    GT5_RF_ERR = 0.025  # meters

    # Return variable
    ret = pd.DataFrame(columns=['TimeUS', 'ofN', 'ofE', 'ofD',
                                'errN', 'errE', 'errD'])

    # Body frame downward velocity according to Rangefinder
    velD = 0  # m/s
    errVelD = 0  # m/s
    rfT = df.TimeUS.iloc[0]  # us

    # Iterating dataframe
    for index, row in df.iterrows():
        if index == 0:
            continue

        # OF rates has updated, calculate NED velocities and error
        if row.ofT != df.ofT.iloc[index - 1]:
            '''
            The segment below calculates the Optical Flow in FRD frame
            and then rotates it into NED frame
            '''
            # Velocity in Bodyframe, [Forward Right Down N E D]
            velBF = np.array([[tan(row.ofY - row.obY)*row.rfD,
                              tan(row.ofX - row.obX)*row.rfD,
                              -velD,
                              0,
                              0,
                              0]])
            # DataFrame to pass to bf_to_ef
            velDF = pd.DataFrame(data=velBF, columns=['ofF', 'ofR', 'ofD',
                                                      'ofN', 'ofE', 'ofD2'])
            # Earth Frame velocities calculated from Optical Flow/Rangefinder
            ef = bf_to_ef(pd.concat([row, velDF.transpose()]).transpose(),
                          ['ofF', 'ofR', 'ofD'],
                          ['ofN', 'ofE', 'ofD2'])
            '''
            The segment below calculates the maximum possible optical flow
            reading given maximum error in IMU and Rangefinder
            '''
            # Rangefinder + max possible error in rangefinder at given height
            errRF = LT5_RF_ERR + row.rfD if (row.rfD < 5) else GT5_RF_ERR + row.rfD
            # Max erroneous velocity in Bodyframe, [Forward Right Down N E D]
            velBF2 = np.array([[tan((row.ofY - row.obY)+GYR_ERR)*errRF,
                              tan((row.ofX - row.obX)+GYR_ERR)*errRF,
                              -errVelD,
                              0,
                              0,
                              0]])
            # Error Dataframe to pass to bf_to_ef
            velDF2 = pd.DataFrame(data=velBF2, columns=['ofF', 'ofR', 'ofD',
                                          'ofN', 'ofE', 'ofD2'])
            errF = bf_to_ef(pd.concat([row, velDF2.transpose()]).transpose(),
                     ['ofF', 'ofR', 'ofD'],
                     ['ofN', 'ofE', 'ofD2'])
            ret = ret.append({'TimeUS': row.TimeUS,
                              'ofN': ef.iloc[0].values[0],
                              'ofE': -ef.iloc[1].values[0], #Flipped a sign
                              'ofD': ef.iloc[2].values[0],
                              'errN': abs(abs(errF.iloc[0].values[0]) - abs(ef.iloc[0].values[0])),
                              'errE': abs(abs(errF.iloc[1].values[0]) - abs(ef.iloc[1].values[0])),
                              'errD': abs(abs(errF.iloc[2].values[0]) - abs(ef.iloc[2].values[0]))},
                             ignore_index=True)
        if row.rfD != df.rfD.iloc[index - 1]:
            velD = (row.rfD - df.rfD.iloc[index - 1]) / ((row.TimeUS - rfT) /
                                                         1000000)  # us -> s
            if row.rfD < 5:
                errVelD = ((row.rfD + LT5_RF_ERR) - df.rfD.iloc[index - 1]) /\
                       ((row.TimeUS - rfT) / 1000000)  # us -> s
            else:
                errVelD = ((row.rfD + GT5_RF_ERR) - df.rfD.iloc[index - 1]) /\
                       ((row.TimeUS - rfT) / 1000000)  # us -> s
            rfT = row.TimeUS
    return(ret)

def AccOF(df, dfOF):
    # Error Constant for LSM303D for Thermo-Mechanical Error, not accounting
    #   for vibrational error introduced by drone
    TM_ACC_ERR = 0.0015 / sqrt(400)  # m/s^2 / sqrt(Hz), where Hz = 400
                                     # From the above, TM_ACC_ERR every 2.5ms
    # Error Constant for L3GD20H
    GYRO_ERR = 0.000191986  # (rad/s/sqrt(Hz)) converted from dps/sqrt(Hz)
    MOT_RAD = .225  # Meters from center of drone to motor
    RMS = sqrt(3)

    # Accumulators
    cumAccVel = np.array([0, 0, 0], dtype=float)
    cumAccVel2 = np.array([0, 0, 0], dtype=float)
    cumAccErr = 0
    cumAccErr2 = 0
    cumGyroErr = 0
    cumGyroErr2 = 0

    # Index iterator for dfOF
    ofIdx = 0
    
    # Return variable
    ret = pd.DataFrame(columns=['TimeUS', 'dAccVel', 'errAcc', 'dOFVel',
                                'errOF'])
    
    for index, row in df.iterrows():
        if index == 0:
            continue
        if ofIdx == (len(dfOF.TimeUS) - 1):
            break

        # dT in seconds since velocity measurements are in meters/second
        dT = (row.aT - df.aT.iloc[index-1])/1000000

        # Accelerometer timestamp has passed Optical Flow, store data
        if row.aT >= dfOF.TimeUS.iloc[ofIdx]:
            ofIdx += 1
            length = norm(cumAccVel)
            ret = ret.append({'TimeUS': row.TimeUS,
                              'dAccVel': length,
                              'errAcc': cumAccErr + (length *
                                                    (1 - cos(cumGyroErr))),
                              'dOFVel': norm([dfOF.ofN.iloc[ofIdx] - dfOF.ofN.iloc[ofIdx - 1],
                                              dfOF.ofE.iloc[ofIdx] - dfOF.ofE.iloc[ofIdx - 1],
                                              dfOF.ofD.iloc[ofIdx] - dfOF.ofD.iloc[ofIdx - 1]]),
                              'errOF': norm([dfOF.errN.iloc[ofIdx],
                                             dfOF.errE.iloc[ofIdx],
                                             dfOF.errD.iloc[ofIdx]]) +
                                       norm([dfOF.ofN.iloc[ofIdx - 1],
                                             dfOF.ofE.iloc[ofIdx - 1],
                                             dfOF.ofD.iloc[ofIdx - 1]])
                              }, ignore_index=True)
            cumAccVel = cumAccVel2
            cumAccVel2 = np.array([0, 0, 0], dtype=float)
            cumAccErr = cumAccErr2
            cumAccErr2 = float(0)
            cumGyroErr = cumGyroErr2
            cumGyroErr2 = float(0)
        # Trapezoidal Integration of accelerometer to get velocity
        if (row.aT/1000) - row.gpT >= 200:
            cumAccVel2[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel2[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel2[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            cumAccErr2 += (RMS * TM_ACC_ERR * (sqrt(1/dT))) * dT
            cumGyroErr2 += GYRO_ERR
        else:
            cumAccVel[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            cumAccErr += (RMS * TM_ACC_ERR * (sqrt(1/dT))) * dT
            cumGyroErr += GYRO_ERR
    return(ret)

def GpsAcc(df, pseudorange=False, gyro=False, NED=False):

    # I'm fairly certain considering the calculation for 3-D rms given by the
    # below link at equation 13:
    # https://gssc.esa.int/navipedia/index.php/Positioning_Error
    # that on each measurement I can simply multiply by sqrt(3) to get the
    # net error from the x, y, and z axis
    # Error Constant for LSM303D for Thermo-Mechanical Error, not accounting
    #   for vibrational error introduced by drone
    TM_ACC_ERR = 0.0015 # m/s^2 / sqrt(Hz), where assuming bandwidth is 100Hz
                        # From the above, TM_ACC_ERR every 2.5ms
    # Error Constant for L3GD20H
    GYRO_ERR = 0.000191986  # (rad/s/sqrt(Hz)) converted from dps/sqrt(Hz)
    MOT_RAD = .225  # Meters from center of drone to motor
    RMS = sqrt(3)

    # Return variable
    if not NED:
        ret = pd.DataFrame(columns=['TimeUS', 'dAccVel', 'errAcc', 'dGpsVel',
                                    'errGps'])
    else:
        ret = pd.DataFrame(columns=['TimeUS', 'dAccVelN', 'dGpsVelN',
                                    'dAccVelE', 'dGpsVelE',
                                    'dAccVelD', 'dGpsVelD',
                                    'errAcc', 'errGpsSA'])
    # Accumulators
    cumAccVel = np.array([0, 0, 0], dtype=float)
    cumAccVel2 = np.array([0, 0, 0], dtype=float)
    cumAccErr = 0
    cumAccErr2 = 0
    cumGyroErr = 0
    cumGyroErr2 = 0

    for index, row in df.iterrows():
        if index == 0:
            continue

        # dT in seconds since velocity measurements are in meters/second
        dT = (row.aT - df.aT.iloc[index-1])/1000000

        # GPS has been updated, store sensor data and refresh
        if row.gpT != df.gpT.iloc[index - 1]:
            length = norm(cumAccVel)
            if NED:
                ret = ret.append({'TimeUS': row.TimeUS,
                                 'dAccVelN': cumAccVel[0],
                                 'dGpsVelN': row.gpN - df.gpN.iloc[index - 1],
                                 'dAccVelE': cumAccVel[1],
                                 'dGpsVelE': row.gpE - df.gpE.iloc[index - 1],
                                 'dAccVelD': cumAccVel[2],
                                 'dGpsVelD': row.gpD - df.gpD.iloc[index - 1],
                                 'errAcc': cumAccErr,
                                 'errGpsSA': row.gpSA + df.gpSA.iloc[index - 1]
                               }, ignore_index=True)
            elif (gyro is True) and (pseudorange is True):
                ret = ret.append({'TimeUS': row.TimeUS,
                                  'dAccVel': length,
                                  'errAcc': cumAccErr + (length *
                                                        (1 - cos(cumGyroErr))),
                                  'dGpsVel': norm([row.gpN-df.gpN.iloc[index - 1],
                                                   row.gpE-df.gpE.iloc[index - 1],
                                                   row.gpD-df.gpD.iloc[index - 1]]),
                                  'errGps': norm([row.gpHA + df.gpHA.iloc[index - 1],
                                                  row.gpVA + df.gpVA.iloc[index - 1]]) /
                                                ((row.gpT - df.gpT.iloc[index - 1])/1000)
                                }, ignore_index=True)
            elif gyro is True:
                ret = ret.append({'TimeUS': row.TimeUS,
                                  'dAccVel': length,
                                  'errAcc': cumAccErr + (length *
                                                        (1 - cos(cumGyroErr))),
                                  'dGpsVel': norm([row.gpN-df.gpN.iloc[index - 1],
                                                   row.gpE-df.gpE.iloc[index - 1],
                                                   row.gpD-df.gpD.iloc[index - 1]]),
                                  'errGps': row.gpSA + df.gpSA.iloc[index - 1]
                                  }, ignore_index=True)
            elif pseudorange is True:
                ret = ret.append({'TimeUS': row.TimeUS,
                                  'dAccVel': length,
                                  'errAcc': cumAccErr,
                                  'dGpsVel': norm([row.gpN-df.gpN.iloc[index - 1],
                                                   row.gpE-df.gpE.iloc[index - 1],
                                                   row.gpD-df.gpD.iloc[index - 1]]),
                                  'errGps': norm([row.gpHA + df.gpHA.iloc[index - 1],
                                                  row.gpVA + df.gpVA.iloc[index - 1]]) /
                                                ((row.gpT - df.gpT.iloc[index - 1])/1000)
                                  }, ignore_index=True)
            else:
                ret = ret.append({'TimeUS': row.TimeUS,
                                  'dAccVel': length,
                                  'errAcc': cumAccErr,
                                  'dGpsVel': norm([row.gpN-df.gpN.iloc[index - 1],
                                                   row.gpE-df.gpE.iloc[index - 1],
                                                   row.gpD-df.gpD.iloc[index - 1]]),
                                  'errGps': row.gpSA + df.gpSA.iloc[index - 1]
                                  }, ignore_index=True)
            cumAccVel = cumAccVel2
            cumAccVel2 = np.array([0, 0, 0], dtype=float)
            cumAccErr = cumAccErr2
            cumAccErr2 = float(0)
            cumGyroErr = cumGyroErr2
            cumGyroErr2 = float(0)
        # Trapezoidal Integration of accelerometer to get velocity
        if (row.aT/1000) - row.gpT >= 200:
            # Accumulate dead-reckoning velocity over each (N,E,D) axis
            cumAccVel2[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel2[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel2[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            # Convert acc/gyro error to dead-reckoning velocity error
            cumAccErr2 += (RMS * TM_ACC_ERR) * dT
            cumGyroErr2 += GYRO_ERR
        else:
            # Accumulate dead-reckoning velocity over each (N,E,D) axis
            cumAccVel[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            # Convert acc/gyro error to dead-reckoning velocity error
            cumAccErr += (RMS * TM_ACC_ERR) * dT
            cumGyroErr += GYRO_ERR
    return(ret)

def GpsAccGC(df):
    
    # I'm fairly certain considering the calculation for 3-D rms given by the
    # below link at equation 13:
    # https://gssc.esa.int/navipedia/index.php/Positioning_Error
    # that on each measurement I can simply multiply by sqrt(3) to get the
    # net error from the x, y, and z axis
    # Error Constant for LSM303D for Thermo-Mechanical Error, not accounting
    #   for vibrational error introduced by drone
    TM_ACC_ERR = 0.0015 * sqrt(400)  # m/s^2 / sqrt(Hz), where Hz = 400
                                     # From the above, TM_ACC_ERR every 2.5ms
    # Error Constant for L3GD20H
    GYRO_ERR = 0.000191986  # (rad/s/sqrt(Hz)) converted from dps/sqrt(Hz)
    RMS = sqrt(3)

    # Return variable
    ret = pd.DataFrame(columns=['TimeUS', 'AccGC', 'errAccGC', 'GpsGC',
                                'errGpsGC'])
    
    # Accumulators
    cumAccVel = np.array([0, 0, 0], dtype=float)
    cumAccVel2 = np.array([0, 0, 0], dtype=float)
    cumAccErr = 0
    cumAccErr2 = 0
    cumGyroErr = 0
    cumGyroErr2 = 0

    for index, row in df.iterrows():
        if index == 0:
            continue

        # dT in seconds since velocity measurements are in meters/second
        dT = (row.aT - df.aT.iloc[index-1])/1000000

        # GPS has been updated, store sensor data and refresh
        if row.gpT != df.gpT.iloc[index - 1]:
            # Calculate the angle between the erroneous vector and ground course
            # vector
            ErrAccGC = np.array([0, 0], dtype=float)
            if cumAccVel[0] > 0:
                ErrAccGC[0] = cumAccVel[0] - cumAccErr
            else:
                ErrAccGC[0] = cumAccVel[0] + cumAccErr
            if cumAccVel[1] > 0:
                ErrAccGC[1] = cumAccVel[1] + cumAccErr
            else:
                ErrAccGC[1] = cumAccVel[1] - cumAccErr
            AccGC = np.array([cumAccVel[0], cumAccVel[1]], dtype=float)/norm(cumAccVel)
            ErrAccGC = ToDeg(np.arccos(np.dot(ErrAccGC/norm(ErrAccGC),
                                              AccGC))) # Magnitude
            AccGC = heading(AccGC)
            # NED Velocities are based on converting geodetic to NED which
            # means our ground course accuracy is based on horizontal accuracy
            # Simply converting recorded NED Velocity to NED Position and using
            # horizontal accuracy for error margins
            p1 = np.array([abs(row.gpN) / abs((row.gpT - df.gpT.iloc[index - 1]) / 1000),
                  abs(row.gpE) / abs((row.gpT - df.gpT.iloc[index - 1]) / 1000)], dtype=float)
            p2 = np.array([p1[0] - (2 * row.gpHA),
                  p1[1] - (2 * row.gpHA)], dtype=float)
            if norm(p1) <= 2 * row.gpHA:
                # if North East distance traveled is too small to overcome 
                # horizontal accuracy, the ground course can't be used
                # confidently for confirmation
                ErrGpsGC = 0
                gpsGC = 1000
            else:
                p1 = p1/norm(p1)
                p2 = p2/norm(p2)
                gpsGC = np.array([row.gpN, row.gpE], dtype=float)
                ErrGpsGC = ToDeg(np.arccos(np.dot(p1, p2))) # Magnitude
                heading(gpsGC)
            # If the error in ground course is greater than 90 degrees, that
            # implies that we could have moved in the opposite direction
            # indicating that the ground course is unusable
            if ErrAccGC >= 90:
                ErrAccGC = 0
                AccGC = -1000
            ret = ret.append({'TimeUS': row.TimeUS,
                              'AccGC': AccGC,
                              'errAccGC': ErrAccGC,
                              'GpsGC': gpsGC,
                              'errGpsGC': ErrGpsGC
                              }, ignore_index=True)

            cumAccVel = cumAccVel2
            cumAccVel2 = np.array([0, 0, 0], dtype=float)
            cumAccErr = cumAccErr2
            cumAccErr2 = float(0)
            cumGyroErr = cumGyroErr2
            cumGyroErr2 = float(0)
        # Trapezoidal Integration of accelerometer to get velocity
        if (row.aT/1000) - row.gpT >= 200:
            # Accumulate dead-reckoning velocity over each (N,E,D) axis
            cumAccVel2[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel2[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel2[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            # Convert acc/gyro error to dead-reckoning velocity error
            cumAccErr2 += (RMS * TM_ACC_ERR) * dT
            cumGyroErr2 += GYRO_ERR
        else:
            # Accumulate dead-reckoning velocity over each (N,E,D) axis
            cumAccVel[0] += ((row.aN + df.aN.iloc[index - 1]) * dT / 2)
            cumAccVel[1] += ((row.aE + df.aE.iloc[index - 1]) * dT / 2)
            cumAccVel[2] += ((row.aDv + df.aDv.iloc[index - 1]) * dT / 2)
            # Convert acc/gyro error to dead-reckoning velocity error
            cumAccErr += (RMS * TM_ACC_ERR) * dT
            cumGyroErr += GYRO_ERR
    return(ret)

def GpsMagGC(df):
    
    # I'm fairly certain considering the calculation for 3-D rms given by the
    # below link at equation 13:
    # https://gssc.esa.int/navipedia/index.php/Positioning_Error
    # that on each measurement I can simply multiply by sqrt(3) to get the
    # net error from the x, y, and z axis
    # Error Constant for LSM303D for Thermo-Mechanical Error, not accounting
    #   for vibrational error introduced by drone
    # Error Constant for L3GD20H
    GYRO_ERR = 0.000191986  # (rad/s/sqrt(Hz)) converted from dps/sqrt(Hz)

    # Return variable
    ret = pd.DataFrame(columns=['TimeUS', 'GpsGC', 'errGpsGC', 'MagGC',
                                'errMagGC'])
    
    # Accumulators
    cumGyroErr = 0
    cumGyroErr2 = 0

    for index, row in df.iterrows():
        if index == 0:
            continue

        # dT in seconds since velocity measurements are in meters/second
        dT = (row.aT - df.aT.iloc[index-1])/1000000

        # GPS has been updated, store sensor data and refresh
        if row.gpT != df.gpT.iloc[index - 1]:
            # gps vector
            gps = np.array([row.gpN, row.gpE],dtype=float)
            # calculating ground course for Magnetometer and GPS
            GpsGC = heading(gps)
            MagGC = ToDeg(np.arctan2(row.m10,row.m00))
            if MagGC < 0:
                MagGC += 360
            # Error based on difference of ground course and erroneous GC
            ErrGps = np.array([abs(row.gpN) - abs(row.gpHA/0.2),
                               abs(row.gpE) + abs(row.gpHA/0.2)],dtype=float)
            ErrGpsGC = ToDeg(np.arccos(np.dot(ErrGps/norm(ErrGps),
                                              abs(gps)/norm(gps))))

            ret = ret.append({'TimeUS': row.TimeUS,
                              'GpsGC': GpsGC,
                              'errGpsGC': ErrGpsGC,
                              'MagGC': MagGC,
                              'errMagGC': ToDeg(cumGyroErr * dT)
                              }, ignore_index=True)

            cumGyroErr = cumGyroErr2
            cumGyroErr2 = float(0)
        # Trapezoidal Integration of accelerometer to get velocity
        if (row.aT/1000) - row.gpT >= 200:
            # Convert gyro error to dead-reckoning angular rate error
            cumGyroErr2 += GYRO_ERR
        else:
            # Convert gyro error to dead-reckoning angular rate error
            cumGyroErr += GYRO_ERR
    return(ret)

def GpsOFGC(df, dfOF):

    # Return variable
    ret = pd.DataFrame(columns=['TimeUS', 'GpsGC', 'errGpsGC', 'OFGC',
                                'errOFGC'])

    # Cumulate OF sensor, [N E]
    cumOFVel = np.array([0, 0], dtype=float)
    cumOFErr = np.array([0, 0], dtype=float)
    
    # Index of ofDF
    ofDFIdx = 0
    
    update = False
    for index, row in df.iterrows():
        if index == 0:
            continue

        # Slowest sensor has updated, flag to store data on next sensor update
        if row.gpT != df.gpT.iloc[index - 1]:
            update = True
            
        # Updating on the fastest sensor to incur the least error due to
        # sync issues.
        if row.ofT != df.ofT.iloc[index - 1]:
            if update:
                # gps vector
                gps = np.array([row.gpN, row.gpE],dtype=float)
                # calculating ground course for optical flow and GPS
                OFGC = heading(cumOFVel)
                GpsGC = heading(gps)
                
                # Error based on difference of ground course and erroneous GC
                ErrOF = np.array([abs(cumOFVel[0]) - abs(cumOFErr[0]),
                               abs(cumOFVel[1]) + abs(cumOFErr[1])],dtype=float)
                ErrOFGC = ToDeg(np.arccos(np.dot(ErrOF/norm(ErrOF),
                                                 abs(cumOFVel)/norm(cumOFVel))))
                ErrGps = np.array([abs(row.gpN) - abs(row.gpHA/0.2),
                                   abs(row.gpE) + abs(row.gpHA/0.2)],dtype=float)
                ErrGpsGC = ToDeg(np.arccos(np.dot(ErrGps/norm(ErrGps),
                                                  abs(gps)/norm(gps))))
                
                ret = ret.append({'TimeUS': row.TimeUS,
                                  'GpsGC': GpsGC,
                                  'errGpsGC': ErrGpsGC,
                                  'OFGC': OFGC,
                                  'errOFGC': ErrOFGC
                                  }, ignore_index=True)
                # Reset update and accumulators when we create a frame
                update = False
                cumOFVel = np.array([0, 0], dtype=float)
                cumOFErr = np.array([0, 0], dtype=float)
            # Always accumulate on an update
            cumOFVel += dfOF[['ofN','ofE']].iloc[ofDFIdx]
            cumOFErr += dfOF[['errN','errE']].iloc[ofDFIdx]
            ofDFIdx += 1
    return(ret)

