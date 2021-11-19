# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:38:16 2021

@author: Bailey K. Srimoungchanh
"""
from Confirmation import confirm, bf_to_ef, norm, reduce_noise
from Confirmation import ToDeg, heading
from functools import reduce
from math import sqrt, cos, tan
import numpy as np
import pandas as pd
from os import mkdir


def process(date, missions, times):
    base = "./Data/" + date
    try:
        mkdir(base + "/Results/")
        mkdir(base + "/Results/Attack/")
        mkdir(base + "/Results/Benign/")
    except:
        print("Results directory already exists.")
    try:
        mkdir(base + "/Results/Attack/GraphData/")
        mkdir(base + "/Results/Benign/GraphData/")
    except:
        print("GraphData directory already exists.")
    try:
        mkdir(base + "/Results/Attack/PairwiseData/")
        mkdir(base + "/Results/Benign/PairwiseData/")
    except:
        print("PairwiseData directory already exists.")

    test_thresholds = 30
    tests = zip(missions, times)
    for name, timing in tests:
        if(len(timing) == 3):
            dir_type = "Attack/"
        else:
            dir_type = "Benign/"

        results = []
        outfile = base + "/Results/" + dir_type + name
        graphData = base + "/Results/" + dir_type + "GraphData/" + name[:-3] + "csv"
        pairwiseData = base + "/Results/" + dir_type + "PairwiseData/" + name[:-4]
        files = []
        first_detected_attack = 0
        for i in range(1,4):
            files.append(base + "/" + date + "-CNF" + str(i) + "-" + name)
        for i in range(1,3):
            files.append(base + "/" + date + "-ACO" + str(i) + "-" + name)
        CNF1 = pd.read_csv(files[0])
        CNF2 = pd.read_csv(files[1])
        CNF3 = pd.read_csv(files[2])
        ACO1 = pd.read_csv(files[3])
        ACO2 = pd.read_csv(files[4])
        CNFs = [CNF1, CNF2, CNF3]
        ACOs = [ACO1, ACO2]
        CNF = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), CNFs)
        ACO = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), ACOs)     
        CNF = CNF.drop(CNF[CNF.TimeUS < timing[0]].index)
        CNF = CNF.drop(CNF[CNF.TimeUS > timing[len(timing) - 1]].index).reset_index(drop=True)
        ACO = ACO.drop(ACO[ACO.TimeUS < timing[0]].index)
        ACO = ACO.drop(ACO[ACO.TimeUS > timing[len(timing) - 1]].index).reset_index(drop=True)
        
        coverages = {"3-Axis":{"ACCOF":{},"ACCGPS":{},"GPSOF":{}},
                     "Net"   :{"ACCOF":{},"ACCGPS":{},"GPSOF":{}},
                     "GC"    :{"GPSMAG":{}, "GPSOF":{}}}
#---Accelerometer and OF---#
        results.append("---Accelerometer and OF---")
    #Velocity Change
    #NED
        results.append("--Tri-Axis Velocity--")
        North = pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],'OF':ACO['COFN']-ACO['POFN'],'OFe':ACO['CNe'] + ACO['PNe'],
                    'Acc':ACO['CAN'],'Acce':ACO['CAe']})
        East = pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],'OF':ACO['COFE']-ACO['POFE'],'OFe':ACO['CNe'] + ACO['PNe'],
                    'Acc':ACO['CAE'],'Acce':ACO['CAe']})
        res1 = confirm(North)
        res2 = confirm(East)
        unions = res1.index.union(res2.index)
        
        
        # Coverage where threshold = 1
        # The CNFs used here is not a typo, we are matching ACO
        # frames to CNF frames for coverage details later
        coverages['3-Axis']['ACCOF'][1] = np.array([0] * len(CNF))
        for i in unions:
            try:
                coverages['3-Axis']['ACCOF'][1][CNF[CNF.TimeUS > ACO.iloc[i].TimeUS].iloc[0].name] = ACO.iloc[i].TimeUS
            except IndexError:
                continue
        res = ACO.iloc[unions]
        
        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=ACO.columns)
        for index, row in ACO.iterrows():
            if ((((row.COFN - row.POFN) <= (row.CNe + row.PNe)) and (row.CAN <= row.CAe)) and 
                (((row.COFE - row.POFE) <= (row.CEe + row.PEe)) and (row.CAE <= row.CAe))):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = ACO[(ACO['TimeUS']>=timing[0]) & 
                              (ACO['TimeUS']<=timing[2])]
            frames_attack = ACO[(ACO['TimeUS']>=timing[1]) & 
                                (ACO['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            indices = []
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")

            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = ACO[(ACO['TimeUS']>=timing[0]) & (ACO['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)
            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))
        
        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = '3-Axis'
        conf_sensors = 'ACCOF'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                # check if index is outside all possible frames
                if res.index[i] > (len(ACO) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if res.index[i+s] == res.index[i]+s:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    frame_time = ACO.iloc[res.index[i+threshold-1]].TimeUS
                    coverages[conf_type][conf_sensors][threshold][CNF[CNF.TimeUS > frame_time].iloc[0].name] = frame_time
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))
        
    #Net
        results.append("--Net Velocity--")
        North = ACO['COFN'] - ACO['POFN']
        East = ACO['COFE'] - ACO['POFE']
        Ne = ACO['CNe'] + ACO['PNe']
        Ee = ACO['CEe'] + ACO['PEe']
        res = confirm(pd.DataFrame(data = {'TimeUS':ACO['TimeUS'],
                                   'OF':(pd.DataFrame(data = {"N":North, "E":East})).apply(norm, axis=1),
                                   'OFe':(pd.DataFrame(data = {"N":Ne, "E":Ee})).apply(norm,axis=1),
                                   'ACC':(ACO[['CAN','CAE']]).apply(norm,axis=1),
                                   'ACCe':sqrt(2)*ACO['CAe']}))
        
        # Coverage where threshold = 1
        coverages['Net']['ACCOF'][1] = np.array([0] * len(CNF))
        for i in res.index:
            try:
                coverages['Net']['ACCOF'][1][CNF[CNF.TimeUS > ACO.iloc[i].TimeUS].iloc[0].name] = ACO.iloc[i].TimeUS
            except IndexError:
                continue
        
        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=ACO.columns)
        for index, row in ACO.iterrows():
            if (norm([row.COFN, row.COFE]) <= norm([row.CNe, row.CEe]) and
                norm([row.CAN, row.CAE]) <= (sqrt(2) * row.CAe)):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = ACO[(ACO['TimeUS']>=timing[0]) & 
                              (ACO['TimeUS']<=timing[2])]
            frames_attack = ACO[(ACO['TimeUS']>=timing[1]) & 
                                (ACO['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
                
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))     
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = ACO[(ACO['TimeUS']>=timing[0]) & (ACO['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))
        
        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = 'Net'
        conf_sensors = 'ACCOF'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                # check if index is outside all possible frames
                if res.index[i] > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if res.index[i+s] == res.index[i]+s:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    frame_time = ACO.iloc[res.index[i+threshold-1]].TimeUS
                    coverages[conf_type][conf_sensors][threshold][CNF[CNF.TimeUS > frame_time].iloc[0].name] = frame_time
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))
        

#---GPS and Magnetometer---#
        results.append("---GPS and Magnetometer---")
        results.append("-Ground Course-")
        #Mag GC
        MagGC = map(ToDeg, map(np.arctan2,CNF['m10'].values,CNF['m00'].values))
        MagGC = [x + 360 if x < 0 else x for x in MagGC]
        
        #GPS GC
        dot = CNF[['CGpN']]
        det = -CNF[['CGpE']]
        GpsGC = map(ToDeg,list(map(np.arctan2,det.values,dot.values)))
        GpsGC = [360 - x if x > 0 else abs(x) for x in GpsGC]
        GpsErr = [0]
        for i in range(len(GpsGC)):
            if i == 0:
                continue
            else:
                GpsErr.append((CNF.iloc[i].CGpe + CNF.iloc[i-1].CGpe)/((CNF.iloc[i].TimeUS-CNF.iloc[i-1].TimeUS)/1000000))
        CNF['GpsErr'] = GpsErr
        ErrGPS = pd.DataFrame(data = {'N':abs(CNF['CGpN']) - abs(CNF['GpsErr']),
                                      'E':abs(CNF['CGpE']) + abs(CNF['GpsErr'])})
        dot = abs(CNF[['CGpN']]).multiply(np.array(ErrGPS['N']), axis=0).add(np.array(abs(CNF[['CGpE']]).multiply(np.array(ErrGPS['E']), axis=0)),axis=0)
        det = abs(CNF[['CGpN']]).multiply(np.array(ErrGPS['E']), axis=0).sub(np.array(abs(CNF[['CGpE']]).multiply(np.array(ErrGPS['N']), axis=0)),axis=0)
        ErrGpsGC = abs(np.array(list(map(ToDeg, map(np.arctan2,det.values,dot.values)))))

        res = confirm(pd.DataFrame( data = {'TimeUS':CNF['TimeUS'],'MagGC':MagGC, 'MagErr':CNF['Gye'],'GPSGC':GpsGC,'GPSErr':ErrGpsGC}),wrap=True)

        # Coverage where threshold = 1
        coverages['GC']['GPSMAG'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['GC']['GPSMAG'][1][CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]] = CNF.iloc[CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]].TimeUS

        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if ((row.CGpN <= row.GpsErr) or (row.CGpE <= row.GpsErr)):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
                

            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])

            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))
       
        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = 'GC'
        conf_sensors = 'GPSMAG'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))

#---Accelerometer and GPS---#
        results.append("---Accelerometer and GPS---")
        #Velocity Change
    #NED
        results.append("--Tri-Axis Velocity--")
        North = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpN']-CNF['PGpN'],'GPe':(CNF['CGpe'] + CNF['PGpe'])/((CNF.iloc[5].TimeUS-CNF.iloc[4].TimeUS)/1000000),
                                     'Acc':CNF['CAN'],'Acce':CNF['CAe']})
        East = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpE']-CNF['PGpE'],'GPe':(CNF['CGpe'] + CNF['PGpe'])/((CNF.iloc[5].TimeUS-CNF.iloc[4].TimeUS)/1000000),
                             'Acc':CNF['CAE'],'Acce':CNF['CAe']})
        Down = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpD']-CNF['PGpD'],'GPe':(CNF['CGpe'] + CNF['PGpe'])/((CNF.iloc[5].TimeUS-CNF.iloc[4].TimeUS)/1000000),
                             'Acc':CNF['CAD'],'Acce':CNF['CAe']})
        res1 = confirm(North)
        res2 = confirm(East)
        res3 = confirm(Down)
        unions = res1.index.union(res2.index).union(res3.index)
        res = CNF.iloc[unions]

        # Coverage where threshold = 1
        coverages['3-Axis']['ACCGPS'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['3-Axis']['ACCGPS'][1][i] = CNF.iloc[i].TimeUS

        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if ((((row.CGpN - row.PGpN) <= (row.CGpe + row.PGpe)) and (row.CAN <= row.CAe)) and 
                (((row.CGpE - row.PGpE) <= (row.CGpe + row.PGpe)) and (row.CAE <= row.CAe)) and
                (((row.CGpD - row.PGpD) <= (row.CGpe + row.PGpe)) and (row.CAD <= row.CAe))):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
            
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))

        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = '3-Axis'
        conf_sensors = 'ACCGPS'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))
                
    #Net
        results.append("--Net Velocity--")
        North = CNF['CGpN'] - CNF['PGpN']
        East = CNF['CGpE'] - CNF['PGpE']
        Down = CNF['CGpD'] - CNF['PGpD']
        res = confirm(pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],
                                   'GPS':(pd.DataFrame(data ={'N':North, 'E':East, 'D':Down})).apply(norm, axis=1),
                                   'GPSe':CNF['GpsErr'],
                                   'ACC':(CNF[['CAN','CAE','CAD']]).apply(norm,axis=1),
                                   'ACCe':sqrt(3)*CNF['CAe']}))
        
        # Coverage where threshold = 1
        coverages['Net']['ACCGPS'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['Net']['ACCGPS'][1][i] = CNF.iloc[i].TimeUS
            
        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if (norm([row.CGpN, row.CGpE, row.CGpD]) <= (sqrt(3) * row.GpsErr) and
                norm([row.CAN, row.CAE, row.CAD]) <= (sqrt(3) * row.CAe)):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
                
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))

        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = 'Net'
        conf_sensors = 'ACCGPS'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))

#---GPS and OF---#
        results.append("---GPS and OF---")
        #Velocity Change
    #NED
        results.append("--Tri-Axis Velocity--")
        North = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpN'],'GPe':CNF['GpsErr'],
                                      'OF':CNF['COFN'],'OFe':CNF['CNe']})
        East = pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],'GPS':CNF['CGpE'],'GPe':CNF['GpsErr'],
                                    'OF':CNF['COFE'],'OFe':CNF['CEe']})
        res1 = confirm(North)
        res2 = confirm(East)
        unions = res1.index.union(res2.index)
        res = CNF.iloc[unions]
        
        # Coverage where threshold = 1
        coverages['3-Axis']['GPSOF'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['3-Axis']['GPSOF'][1][i] = CNF.iloc[i].TimeUS
            
        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if (((row.CGpN <= row.GpsErr) and (row.COFN <= row.CNe)) and 
                ((row.CGpE <= row.GpsErr) and (row.COFE <= row.CEe))):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
            
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = frames_test.index.difference(pd.merge(frames_test, invalid, how='inner', on=['TimeUS']).index)
            frames_detected_test_valid = frames_detected_test.index.difference(pd.merge(invalid, frames_detected_test, how='inner', on=['TimeUS']).index)
            indices = frames_detected_test_valid
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))

        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))

        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")

        # Coverage testing of thresholds
        conf_type = '3-Axis'
        conf_sensors = 'GPSOF'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))

    #Net
        results.append("--Net Velocity--")
        res = confirm(pd.DataFrame(data = {'TimeUS':CNF['TimeUS'],
                                           'OF':(CNF[['COFN','COFE']]).apply(norm, axis=1),
                                           'OFe':(CNF[['CNe','CEe',]]).apply(norm,axis=1),
                                           'GPS':(CNF[['CGpN','CGpE']]).apply(norm,axis=1),
                                           'GPSe':sqrt(2)*CNF['CGpe']/((CNF.iloc[5].TimeUS-CNF.iloc[4].TimeUS)/1000000.0)}))
                                                                                      # I selected two arbitrary points to get the dT
        
        # Coverage where threshold = 1
        coverages['Net']['GPSOF'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['Net']['GPSOF'][1][i] = CNF.iloc[i].TimeUS

        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if (norm([row.COFN, row.COFE]) <= norm([row.CNe, row.CEe]) and
                norm([row.CGpN, row.CGpE]) <= (sqrt(2) * row.CGpe)):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
            
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = frames_test.index.difference(pd.merge(frames_test, invalid, how='inner', on=['TimeUS']).index)
            frames_detected_test_valid = frames_detected_test.index.difference(pd.merge(invalid, frames_detected_test, how='inner', on=['TimeUS']).index)
            indices = frames_detected_test_valid
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))

        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))
        
        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
    
        # Coverage testing of thresholds
        conf_type = 'Net'
        conf_sensors = 'GPSOF'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))

    #Ground Course
        results.append("-Ground Course-")
        dot = CNF[['COFN']]
        det = -CNF[['COFE']]
        OFGC = map(ToDeg, map(np.arctan2, det.values, dot.values))
        OFGC = [360 - x if x > 0 else abs(x) for x in OFGC]
        ErrOF = pd.DataFrame(data = {'N':abs(CNF['COFN']) - CNF['CNe'],
                                     'E':abs(CNF['COFE']) + CNF['CEe']})
        dot = abs(CNF[['COFN']]).multiply(np.array(ErrOF['N']), axis=0).add(np.array(abs(CNF[['COFE']]).multiply(np.array(ErrOF['E']), axis=0)),axis=0)
        det = abs(CNF[['COFN']]).multiply(np.array(ErrOF['E']), axis=0).sub(np.array(abs(CNF[['COFE']]).multiply(np.array(ErrOF['N']), axis=0)),axis=0)
        ErrOFGC = abs(np.array(list(map(ToDeg, map(np.arctan2, det.values, dot.values)))))
        
        res = confirm(pd.DataFrame( data = {'TimeUS':CNF['TimeUS'],
                                            'OFGC':OFGC,
                                            'OFErr':ErrOFGC,
                                            'GPSGC':GpsGC,
                                            'GPSErr':ErrGpsGC}),
                      wrap=True)
        
        # Coverage where threshold = 1
        coverages['GC']['GPSOF'][1] = np.array([0] * len(CNF))
        for i in res.index:
            coverages['GC']['GPSOF'][1][CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]] = CNF.iloc[CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]].TimeUS

        #Separating frames that are not useful for confirmation
        invalid = pd.DataFrame(columns=CNF.columns)
        for index, row in CNF.iterrows():
            if ((norm([row.CGpN, row.CGpE]) <= (sqrt(2) * row.GpsErr)) and
                (norm([row.COFN, row.COFE]) <= (norm([row.CNe, row.CEe])))):
                    invalid = invalid.append(row)
            else:
                continue
        
        #Calculating Frames for Results
        if len(timing) == 3: #Attack Results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[2])]
            frames_detected_attack = res[(res['TimeUS']>=timing[1]) & 
                                         (res['TimeUS']<=timing[2])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & 
                              (CNF['TimeUS']<=timing[2])]
            frames_attack = CNF[(CNF['TimeUS']>=timing[1]) & 
                                (CNF['TimeUS']<=timing[2])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_attack_valid = np.setdiff1d(frames_attack.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            frames_detected_attack_valid = np.setdiff1d(frames_detected_attack.TimeUS, invalid.TimeUS)
            if len(frames_detected_attack) != 0:
                first_detected_attack = frames_detected_attack.iloc[0]
            indices = []
            for i in frames_detected_attack_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_attack_valid = len(frames_attack_valid)
            frames_detected_attack_valid = len(frames_detected_attack_valid)
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_detected_test = len(frames_detected_test)
            frames_detected_attack = len(frames_detected_attack)
            frames_test = len(frames_test)
            frames_attack = len(frames_attack)

            #Save frame results
            results.append("Detected Test: " + 
                           str(frames_detected_test) + "/" + str(frames_test) +
                           "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_attack == 0:
                results.append("Detected Attack: Spoofing Limit reached on start.")
            else:
                results.append("Detected Attack:" + 
                               str(frames_detected_attack) + "/" + str(frames_attack) +
                               "(" + str(round(frames_detected_attack * 100/frames_attack, 2)) + "%)")
            if frames_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Test without Invalid: " + 
                               str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                               "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
            if frames_attack_valid == 0:
                results.append("All Attack frames were invalid.")
            else:
                results.append("Detected Attack without Invalid:" + 
                               str(frames_detected_attack_valid) + "/" + str(frames_attack_valid) +
                               "(" + str(round(frames_detected_attack_valid * 100/frames_attack_valid, 2)) + "%)")
            
            FP = frames_detected_test - frames_detected_attack
            TN = frames_test - frames_attack
            TP = frames_detected_attack
            FN = frames_attack - frames_detected_attack
            FP_s = frames_detected_test_valid - frames_detected_attack_valid
            TN_s = frames_test_valid - frames_attack_valid
            TP_s = frames_detected_attack_valid
            FN_s = frames_attack_valid - frames_detected_attack_valid
            if((FP_s+TN_s) == 0):
                results.append("FPR(Strict): N/A")
            else:
                results.append("FPR(Strict): " + str(round(FP_s/(FP_s+TN_s) * 100,2)))
            if((FP+TN) == 0):
                results.append("FPR(Permissive): N/A")
            else:
                results.append("FPR(Permissive): " + str(round(FP/(FP+TN) * 100,2)))
            if((TP_s+FN_s) == 0):
                results.append("TPR(Strict): N/A")
            else:
                results.append("TPR(Strict): " + str(round(TP_s/(TP_s+FN_s) * 100,2)))
            if((TP+FN) == 0):
                results.append("TPR(Permissive): N/A")
            else:
                results.append("TPR(Permissive): " + str(round(TP/(TP+FN) * 100,2)))
        else: #Benign results
            frames_detected_test = res[(res['TimeUS'] >=timing[0]) & 
                                          (res['TimeUS']<=timing[1])]
            frames_test = CNF[(CNF['TimeUS']>=timing[0]) & (CNF['TimeUS']<=timing[1])]
            frames_test_valid = np.setdiff1d(frames_test.TimeUS, invalid.TimeUS)
            frames_detected_test_valid = np.setdiff1d(frames_detected_test.TimeUS, invalid.TimeUS)
            indices = []
            for i in frames_detected_test_valid:
                indices.append(res.loc[res['TimeUS'] == i].index[0])
            
            #Just need the length of the dataframes above
            frames_test_valid = len(frames_test_valid)
            frames_detected_test_valid = len(frames_detected_test_valid)
            frames_test = len(frames_test)
            frames_detected_test = len(frames_detected_test)

            results.append("Detected Overall (FPR): " + 
                            str(frames_detected_test) + "/" + str(frames_test) +
                            "(" + str(round(frames_detected_test * 100/frames_test, 2)) + "%)")
            if frames_detected_test_valid == 0:
                results.append("No detected frames in Test Window")
            else:
                results.append("Detected Overall without Invalid (FPR): " + 
                                str(frames_detected_test_valid) + "/" + str(frames_test_valid) +
                                "(" + str(round(frames_detected_test_valid * 100/frames_test_valid, 2)) + "%)")
        results.append("Valid Frames (%): " + str(frames_test_valid * 100/frames_test))
        
        streak = 0
        counter = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                counter += 1
            else:
                if counter > streak:
                    streak = counter
                counter = 1
        if counter > streak:
            streak = counter
        if len(indices) == 0:
            results.append("Streak N/A")
            results.append("TTD (Strict) N/A")
        elif len(timing) == 3:
            results.append("Time-To-Detection (Strict): " + str((res.loc[indices[0]].TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("Streak of Windows: " + str(streak))
            
        if(( len(timing) == 3) and (frames_detected_attack != 0)):
            results.append("Time-To-Detection (Permissive): " + str((first_detected_attack.TimeUS - timing[1])/1000) + "ms")
        else:
            results.append("TTD (Permissive) N/A")
            
        # Coverage testing of thresholds
        conf_type = 'GC'
        conf_sensors = 'GPSOF'
        threshold = 2
        for x in range(2, test_thresholds+1):
            seq = [x for x in range(1,threshold)]
            coverages[conf_type][conf_sensors][threshold] = np.array([0] * (len(CNF)-threshold+1))
            for i in range(len(res.index)):
                counter = 0
                CNF_index = CNF.index[CNF.TimeUS == res.iloc[i].TimeUS]
                # check if index is outside all possible frames
                if CNF_index > (len(CNF) - threshold + 1):
                    break
                # check if enough frames are left to confirm
                if (len(res.index) - (i+threshold-1)) < threshold:
                    break
                for s in seq:
                    if CNF_index+s == CNF.index[CNF.TimeUS == res.iloc[i+s].TimeUS]:
                        counter += 1
                # Marks a detected frame
                if counter == len(seq):
                    coverages[conf_type][conf_sensors][threshold][CNF_index+threshold-1] = CNF.iloc[CNF_index+threshold-1].TimeUS
            if(np.count_nonzero(coverages[conf_type][conf_sensors][threshold]) == 0):
                break
            threshold += 1
        # gaurantees an empty coverage when threshold of 2 detects nothing
        if len(coverages[conf_type][conf_sensors]) == 1:
            coverages[conf_type][conf_sensors][2] = np.array([0] * (len(CNF)-1))

        with open(outfile, 'w') as of:
            of.writelines(line + '\n' for line in results)
            
        # ROC information
        if len(timing) == 2:
            triaxis = []
            net = []
            gc = []
            # Benign data only cares about FPR

            #3-Axis processing
            longest = max(len(item) for item in coverages['3-Axis'].values())
            for i in range(1,longest+1):
                inter = np.array([0] * (len(CNF)-i+1))
                for key, value in coverages['3-Axis'].items():
                    if len(value) < i:
                        continue
                    inter = np.where(inter ==0, value[i], inter)
                triaxis = np.append(triaxis, np.count_nonzero(inter)/len(inter))
                
            #Net processing
            longest = max(len(item) for item in coverages['Net'].values())
            for i in range(1,longest+1):
                inter = np.array([0] * (len(CNF)-i+1))
                for key, value in coverages['Net'].items():
                    if(name.startswith("C-")):
                        if(key == "GPSOF"): #Copter does not use GPSOF in Net
                            continue
                    elif(name.startswith("P-")):
                        if(key == "ACCGPS"): #Plane does not use ACCGPS in Net
                            continue
                    if len(value) < i:
                        continue
                    inter = np.where(inter ==0, value[i], inter)
                net = np.append(net, np.count_nonzero(inter)/len(inter))
                
            #GC processing
            longest = max(len(item) for item in coverages['GC'].values())
            for i in range(1,longest+1):
                inter = np.array([0] * (len(CNF)-i+1))
                for key, value in coverages['GC'].items(): #Use both GPSMAG and GPSOF
                    if len(value) < i:
                        continue
                    inter = np.where(inter ==0, value[i], inter)
                gc = np.append(gc, np.count_nonzero(inter)/len(inter))
                
            # Match array lengths then output the csv
            gc = np.pad(gc, (0, test_thresholds - len(gc)), 'constant')
            net = np.pad(net, (0, test_thresholds - len(net)), 'constant')
            triaxis = np.pad(triaxis, (0, test_thresholds - len(triaxis)), 'constant')
            pd.DataFrame(data={'THR':list(range(1,len(net)+1)),
                               'Frames':np.flip(np.array(list(range(len(CNF)-len(gc)+1,len(CNF)+1)))),
                               'Tri-Axis FPR':triaxis,
                               'Net FPR': net,
                               'GC FPR':gc}).to_csv(graphData,index=False)
            
            # Output pairwise FPR data
            # Net, 3-Axis, and GC
            suffixes = ['Net','3-Axis','GC']
            outFiles = [pairwiseData + '-' + suffix + '.csv' for suffix in suffixes]
            for i in range(3):
                ACCOF = np.array([], dtype=float)
                GPSOF = np.array([], dtype=float)
                ACCGPS = np.array([], dtype=float)
                GPSMAG = np.array([], dtype=float)
                for key, value in coverages[suffixes[i]].items():
                    for key2, value2 in value.items():
                        if suffixes[i] != "GC":
                            if key == 'ACCOF':
                                ACCOF = np.append(ACCOF, np.count_nonzero(value2)/len(value2))
                            elif key == 'GPSOF':
                                GPSOF = np.append(GPSOF, np.count_nonzero(value2)/len(value2))
                            elif key == 'ACCGPS':
                                ACCGPS = np.append(ACCGPS, np.count_nonzero(value2)/len(value2))
                        else:
                            if key == 'GPSMAG':
                                GPSMAG = np.append(GPSMAG, np.count_nonzero(value2)/len(value2))
                            elif key == 'GPSOF':
                                GPSOF = np.append(GPSOF, np.count_nonzero(value2)/len(value2))
                            
                # Match array lengths then output the csv
                GPSOF = np.pad(GPSOF, (0, test_thresholds - len(GPSOF)), 'constant')
                ACCOF = np.pad(ACCOF, (0, test_thresholds - len(ACCOF)), 'constant')
                ACCGPS = np.pad(ACCGPS, (0, test_thresholds - len(ACCGPS)), 'constant')
                GPSMAG = np.pad(GPSMAG, (0, test_thresholds - len(GPSMAG)), 'constant')
                
                if suffixes[i] != "GC":
                    outCsv = pd.DataFrame(data={'Threshold': range(1,test_thresholds+1),
                                           'ACCOF(FPR)':ACCOF,
                                           'GPSOF(FPR)':GPSOF,
                                           'ACCGPS(FPR)':ACCGPS})
                else:
                    outCsv = pd.DataFrame(data={'Threshold': range(1,test_thresholds+1),
                                           'GPSOF(FPR)':GPSOF,
                                           'GPSMAG(FPR)':GPSMAG})
                outCsv.to_csv(outFiles[i], index=False)          
            
        elif len(timing) == 3:
            net = []
            gc = []
            # Adversarial data needs to calculate FPR and TPR for every THR 
            # Also needs to combine Net and GC data
            #Net processing
            longest = max(len(item) for item in coverages['Net'].values())
            for i in range(1,longest+1):
                inter = np.array([0] * (len(CNF)-i+1))
                for key, value in coverages['Net'].items():
                    if(name.startswith("C-")):
                        if(key == "GPSOF"): #Copter does not use GPSOF in Net
                            continue
                    elif(name.startswith("P-")):
                        if(key == "ACCGPS"): #Plane does not use ACCGPS in Net
                            continue 
                    if len(value) < i:
                        continue
                    inter = np.where(inter ==0, value[i], inter)
                net.append(inter)
            
            #GC processing
            longest = max(len(item) for item in coverages['GC'].values())
            for i in range(1,longest+1):
                inter = np.array([0] * (len(CNF)-i+1))
                for key, value in coverages['GC'].items():
                    if len(value) < i:
                        continue
                    inter = np.where(inter ==0, value[i], inter)
                gc.append(inter)
            
            # Match array lengths then output the csv
            if len(net) > len(gc):
                for i in range(len(gc),len(net)):
                    gc.append(np.array([0]*len(net[i])))                 
            elif len(gc) > len(net):
                for i in range(len(net),len(gc)):
                    net.append(np.array([0]*len(gc[i])))
            gc = np.array(gc,dtype=object)
            net = np.array(net,dtype=object)
            
            # Storing combined results in GC array
            for i in range(len(gc)):
                gc[i] = np.where(gc[i] == 0, net[i], gc[i])
                
            # Calculate system-wise FPR and TPR
            FPR = np.array([], dtype=float)
            TPR = np.array([], dtype=float)
            TTD = np.array([0] * len(gc))
            for i in range(len(gc)): # Number of thresholds
                FP = 0
                FN = 0
                TP = 0
                TN = 0
                for s in range(len(gc[i])): # Frames in threshold array
                    if TTD[i] == 0 and gc[i][s] >= timing[1]:
                        TTD[i] = (gc[i][s] - float(timing[1]))/1000.0
                    # False Negative check
                    if CNF.iloc[s].TimeUS >= timing[1]:
                        if gc[i][s] == 0:
                            FN += 1
                        else:
                            TP += 1
                    # False Positive check
                    elif CNF.iloc[s].TimeUS < timing[1]:
                        if gc[i][s] != 0:
                            FP += 1
                        else:
                            TN += 1
                if FP + TN == 0:
                    FPR = np.append(FPR, -1)
                else:
                    FPR = np.append(FPR, FP/(FP+TN))
                if TP + FN == 0:
                    # I have only seen this occur when the stealthy attack has no room
                    TPR = np.append(TPR, -1)
                else:
                    TPR = np.append(TPR, TP/(TP+FN))
                
            pd.DataFrame(data={'THR':list(range(1,len(gc)+1)),
                               'Frames':np.flip(np.array(list(range(len(CNF)-len(gc)+1,len(CNF)+1)))),
                               'FPR': FPR,
                               'TPR': TPR,
                               'TTD': TTD}).to_csv(graphData,index=False)
            
            # Output pairwise FPR and TPR Data
            # Net, 3-Axis, and GC
            suffixes = ['Net','3-Axis','GC']
            outFiles = [pairwiseData + '-' + suffix + '.csv' for suffix in suffixes]
            frames_benign = frames_test - frames_attack
            for i in range(3):
                FPR_ACCOF = np.array([], dtype=float)
                FPR_GPSOF = np.array([], dtype=float)
                FPR_ACCGPS = np.array([], dtype=float)
                FPR_GPSMAG = np.array([], dtype=float)
                TPR_ACCOF = np.array([], dtype=float)
                TPR_GPSOF = np.array([], dtype=float)
                TPR_ACCGPS = np.array([], dtype=float)
                TPR_GPSMAG = np.array([], dtype=float)
                for key, value in coverages[suffixes[i]].items():
                    for key2, value2 in value.items():
                        frames_attack = len(value2) - frames_benign
                        #FPR
                        FPR = np.where((value2 >= timing[0]) & (value2 <= timing[1]), 1, 0).sum()
                        #TPR
                        TPR = np.where((value2 >= timing[1]) & (value2 <= timing[2]), 1, 0).sum()
                        if(frames_attack == 0):
                            frames_attack = 1
                            TPR = -1
                        if(suffixes[i] != "GC"):
                            if key == 'ACCOF':
                                FPR_ACCOF = np.append(FPR_ACCOF, FPR/frames_benign)
                                TPR_ACCOF = np.append(TPR_ACCOF, TPR/frames_attack)
                            elif key == 'GPSOF':
                                FPR_GPSOF = np.append(FPR_GPSOF, FPR/frames_benign)
                                TPR_GPSOF = np.append(TPR_GPSOF, TPR/frames_attack)
                            elif key == 'ACCGPS':
                                FPR_ACCGPS = np.append(FPR_ACCGPS, FPR/frames_benign)
                                TPR_ACCGPS = np.append(TPR_ACCGPS, TPR/frames_attack)
                        else:
                            if key == 'GPSMAG':
                                FPR_GPSMAG = np.append(FPR_GPSMAG, FPR/frames_benign)
                                TPR_GPSMAG = np.append(TPR_GPSMAG, TPR/frames_attack)
                            elif key == 'GPSOF':
                                FPR_GPSOF = np.append(FPR_GPSOF, FPR/frames_benign)
                                TPR_GPSOF = np.append(TPR_GPSOF, TPR/frames_attack)
                # Match array lengths then output the csv
                FPR_ACCOF = np.pad(FPR_ACCOF, (0, test_thresholds - len(FPR_ACCOF)), 'constant')
                TPR_ACCOF = np.pad(TPR_ACCOF, (0, test_thresholds - len(TPR_ACCOF)), 'constant')
                FPR_ACCGPS = np.pad(FPR_ACCGPS, (0, test_thresholds - len(FPR_ACCGPS)), 'constant')
                TPR_ACCGPS = np.pad(TPR_ACCGPS, (0, test_thresholds - len(TPR_ACCGPS)), 'constant')
                FPR_GPSOF = np.pad(FPR_GPSOF, (0, test_thresholds - len(FPR_GPSOF)), 'constant')
                TPR_GPSOF = np.pad(TPR_GPSOF, (0, test_thresholds - len(TPR_GPSOF)), 'constant')
                FPR_GPSMAG = np.pad(FPR_GPSMAG, (0, test_thresholds - len(FPR_GPSMAG)), 'constant')
                TPR_GPSMAG = np.pad(TPR_GPSMAG, (0, test_thresholds - len(TPR_GPSMAG)), 'constant')

                if suffixes[i] != "GC":
                    outCsv = pd.DataFrame(data={'Threshold': range(1,test_thresholds+1),
                                           'ACCOF(FPR)':FPR_ACCOF,
                                           'ACCOF(TPR)':TPR_ACCOF,
                                           'GPSOF(FPR)':FPR_GPSOF,
                                           'GPSOF(TPR)':TPR_GPSOF,
                                           'ACCGPS(FPR)':FPR_ACCGPS,
                                           'ACCGPS(TPR)':TPR_ACCGPS})
                else:
                    outCsv = pd.DataFrame(data={'Threshold': range(1,test_thresholds+1),
                                           'GPSOF(FPR)':FPR_GPSOF,
                                           'GPSOF(TPR)':TPR_GPSOF,
                                           'GPSMAG(FPR)':FPR_GPSMAG,
                                           'GPSMAG(TPR)':TPR_GPSMAG})
                
                outCsv.to_csv(outFiles[i], index=False)        

def main():
    date = "2021-11-17"
    missions = [
                # "C-Motion-NEO-1cm.txt",
                "C-Motion-NEO-250cm.txt"
                # "C-Motion-ZED-1cm.txt","C-Motion-ZED-250cm.txt",
                # "C-Idle-NEO-1cm.txt","C-Idle-NEO-250cm.txt",
                # "C-Idle-ZED-1cm.txt","C-Idle-ZED-250cm.txt",
                # "C-Stealth-NEO.txt","C-Stealth-ZED.txt",
                # "C-Circle-NEO.txt","C-Circle-ZED.txt",
                # "C-Square-NEO.txt","C-Square-ZED.txt",
                # "C-Wave-NEO.txt","C-Wave-ZED.txt"
                # "P-Motion-NEO-1cm.txt","P-Motion-NEO-250cm.txt",
                # "P-Motion-ZED-1cm.txt","P-Motion-ZED-250cm.txt",
                # "P-Stealth-ZED.txt", "P-Stealth-NEO.txt",
                # "P-Circle-NEO.txt","P-Circle-ZED.txt",
                # "P-Square-NEO.txt","P-Square-ZED.txt",
                # "P-Wave-NEO.txt","P-Wave-ZED.txt"
                ]
    # Motion formatted as [Mission: 2 WP, Enabled Attack, Disabled Attack]
    # Idle formatted as [Mode Guided, Enabled Attack, Disabled Attack]
    # Square and Wave formatted as  [Mission: 2 WP, RTL]
    # Copter Circle formatted as [Mode Circle, Mode RTL]
    # Plane Circle formatted as [Sim Delay, Disarm]
    # Stealth formatted as [Altitude Reached, Enabled Attacked, Disabled Attack/Attack Limit]
    times = [
                # [58405795,71458905,132086311], 
                [58405795,71458905,132086311],
                # [58405795,71458905,132086311], [58405795,71458905,132086311],
                # [52030846,73081589,133883925], [52030846,73081589,133883925],
                # [52030846,73081589,133883925], [52030846,73081589,133883925],
                # [52030846,58080925,118658351], [52030846,58080925,90203904],
                # [52030846,174028694], [52030846,174028694],
                # [66836588,157028830], [66836588,157028830],
                # [61403762,131025902], [61403762,131025902]
                # [28300342,47280247,107580284], [28300342,47280247,107580284],
                # [28300342,47280247,107580284], [28300342,47280247,107580284],
                # [29800575,48180720,48200712],[30100455,48180720,48200712],
                # [21560539,261560334], [21560539,261560334],
                # [28300342,162200927], [28300342,162200927],
                # [33800641,120120266], [33800641,120120266]
            ]
    process(date, missions, times)

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

