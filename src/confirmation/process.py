import pandas as pd
from functools import reduce
from math import atan, degrees
#TQDM for progress information
from tqdm.notebook import trange

#Wrapper to convert magy and magx series to heading
def mag_to_heading(z, y, x):
    res = []
    for index in range(len(y)):
        norm = z[index] + y[index] + x[index]
        norm_x = x[index]/norm
        norm_y = y[index]/norm
        if x[index] > 0:
            res.append(90 - degrees(atan(norm_y/norm_x)))
        elif x[index] < 0:
            res.append(270 - degrees(atan(norm_y/norm_x)))
        else:
            if y[index] < 0:
                res.append(180)
            elif y[index] > 0:
                res.append(0)
    return(pd.Series(res, name="Mag Heading"))

#Wrapper to convert magz and magy series to pitch
def mag_to_pitch(z, y, x):
    res = []
    for index in range(len(z)):
        norm = z[index] + y[index] + x[index]
        norm_z = z[index]/norm
        norm_x = x[index]/norm
        if x[index] > 0:
            res.append(90 - degrees(atan(-norm_z/norm_x)))
        elif x[index] < 0:
            res.append(270 - degrees(atan(-norm_z/norm_x)))
        else:
            if z[index] < 0:
                res.append(0)
            elif z[index] > 0:
                res.append(180)
    return(pd.Series(res, name="Mag Pitch"))

#Open log file and parse out the SNS data into pandas dataframes
def process_SNS(path, out="", SNS_COUNT=4):
    SNSs = [{} for i in range(SNS_COUNT)]
    SNSs_counter = [0] * SNS_COUNT
    cols = [None] * SNS_COUNT

    with open(path) as infile:
        number_lines = sum(1 for line in open(path))
        for counter in trange(number_lines):
            line = infile.readline()
            strippedLine = line.strip()

            if "FMT" in strippedLine:
                for i in range(1, SNS_COUNT+1):
                    if ("SNS" + str(i)) in strippedLine:
                        cols[i-1] = [val.strip() for val in strippedLine.split(',')[5:]]
                        break
            else:
                for i in range(1, SNS_COUNT+1):
                    if ("SNS" + str(i)) in strippedLine:
                        SNSs[i-1][SNSs_counter[i-1]] = dict(zip(cols[i-1], [val.strip() for val in strippedLine.split(',')[1:]]))
                        SNSs_counter[i-1] = SNSs_counter[i-1] + 1
                        break
        
        for i in range(SNS_COUNT):
            SNSs[i] = pd.DataFrame.from_dict(SNSs[i], "index")

    SNS = reduce(lambda left, right: pd.merge(left, right, on='TimeUS'), SNSs)

    if out:
        SNS.to_csv(out, index=False)
    return(SNS)

#Trapezoidal Integration
def trap_integrate(ts, signal):
    result = []
    for val in range(1,len(signal)):
        height = (signal[val] + signal[val-1]) / 2
        dt = ts[val] - ts[val-1]
        result.append(height * dt)
    return(pd.Series(result, name=signal.name + "_int", dtype=signal.dtype))