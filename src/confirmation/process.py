import pandas as pd
from functools import reduce
from math import atan, degrees
from pymap3d import geodetic2enu
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

#leaky integrator for pandas Series data
def leaky_integrator(signal, alpha=1):
    filtered = [signal[0]]
    val = signal[0]
    for sample in signal[1:]:
        val += ((sample-val) * alpha)
        filtered.append(val)
    return(pd.Series(filtered, name=signal.name + "_leaky"))

#Trapezoidal Integration
def trap_integrate(ts, signal):
    result = []
    for val in range(1,len(signal)):
        height = (signal[val] + signal[val-1]) / 2
        dt = ts[val] - ts[val-1]
        result.append(height * dt)
    return(pd.Series(result, name=signal.name + "_int", dtype=signal.dtype))

#Change in signal
def change_in_signal(signal):
    result = []
    for val in range(1,len(signal)):
        result.append(signal[val] - signal[val-1])
    return(pd.Series(result, name=signal.name + "_dt", dtype=signal.dtype))

def geodetic2ned(lat, lng, alt, lat0=0, lng0=0, alt0=0):
    #if lat0=lng0=alt0=0, assume first row is origin
    if lat0==0:
        lat0 = lat[0]
    if lng0==0:
        lng0 = lng[0]
    if alt0==0:
        alt0 = alt[0]
    local = [df.lat[0], df.lng[0], df.gpAlt[0]]
    target = [df.lat[100], df.lng[100], df.gpAlt[100]]
    res = geodetic2enu( target[0], target[1], target[2], local[0], local[1], local[2])
    res = [res[1], res[0], -res[2]]