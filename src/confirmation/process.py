import pandas as pd
from functools import reduce
from math import atan, degrees, sqrt
from pymap3d import geodetic2enu
from numpy.polynomial.polynomial import Polynomial
#TQDM for progress information
from tqdm.notebook import trange
from IPython.display import clear_output

#Gets absolute difference in values.
#Only added to diff degree values with modulo arithmetic
def diff(x, y, wrap=False):
    if not wrap:
        return(abs(x - y))
    else:
        res = []
        for index in range(len(x)):
            if (abs(x[index] - y[index]) >= 180):
                res.append(abs(x[index] - (y[index] + 360)))
            else:
                res.append(abs(x[index] - y[index]))
        return(pd.Series(res))

#Wrapper to convert magy and magx series to heading
def mag_to_heading(z, y, x):
    res = []
    for index in trange(len(y), desc="mag_to_heading"):
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
    clear_output()
    return(pd.Series(res, name="Mag Heading"))

#Wrapper to convert magz and magy series to pitch
def mag_to_pitch(z, y, x):
    res = []
    for index in trange(len(z), desc="mag_to_pitch"):
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
    clear_output()
    return(pd.Series(res, name="Mag Pitch"))

#Open log file and parse out the SNS data into pandas dataframes
def process_SNS(path, out="", SNS_COUNT=4):
    SNSs = [{} for i in range(SNS_COUNT)]
    SNSs_counter = [0] * SNS_COUNT
    cols = [None] * SNS_COUNT

    with open(path) as infile:
        number_lines = sum(1 for line in open(path))
        for counter in trange(number_lines, "process_SNS"):
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

#Takes low frequency information in signal, IIR
def low_pass_filter(signal, alpha):
    filtered = [signal[0]]
    val = signal[0]
    for sample in signal[1:]:
        val += ((sample-val) * alpha)
        filtered.append(val)
    return(pd.Series(filtered, name=signal.name + "_lpf"))

#Take high frequency information in signal, IIR
def high_pass_filter(signal, alpha):
    res = signal - low_pass_filter(signal, 1-alpha)
    return(pd.Series(res, name=signal.name + "_hpf"))

#Trapezoidal Integration
def trap_integrate(ts, signal):
    result = []
    for val in trange(1,len(signal), desc="trap_integrate"):
        height = (signal[val] + signal[val-1]) / 2
        dt = ts[val] - ts[val-1]
        result.append(height * dt)
    clear_output()
    return(pd.Series(result, name=signal.name + "_int", dtype=signal.dtype))

#Change in signal
def change_in_signal(signal):
    result = []
    for val in trange(1,len(signal), desc="change_in_signal"):
        result.append(signal[val] - signal[val-1])
    try:
        clear_output()
        return(pd.Series(result, name=signal.name + "_dt", dtype=signal.dtype))
    except (TypeError, AttributeError):
        return(result)

def body_to_earth2D(front, right, m00, m10):
    if len(front) != len(right) or len(right) != len(m00) or len(m00) != len(m10):
        print("Mismatched series: %f, %f, %f, %f" % (len(front), len(right), len(m00), len(m10)))
        return([])
    
    normalized = []
    for index in trange(len(m00), desc="Normalizing"):
        normalized.append([m00[index]/sqrt(m00[index]**2+m10[index]**2), m10[index]/sqrt(m00[index]**2+m10[index]**2)])
    clear_output()
    res = []
    for index in trange(len(normalized), desc="Rotating"):
        res.append([front[index] * normalized[index][0] + right[index] * normalized[index][1],
                    -front[index] * normalized[index][1] + right[index] * normalized[index][0]])
    clear_output()
    return(pd.DataFrame(res, columns=["North", "East"]))

def earth_to_body2D(north, east, m00, m10):
    if len(north) != len(east) or len(east) != len(m00) or len(m00) != len(m10):
        print("Mismatched series: %f, %f, %f, %f" % (len(north), len(east), len(m00), len(m10)))
        return([])
    
    normalized = []
    for index in trange(len(m00), desc="Normalizing"):
        normalized.append([m00[index]/sqrt(m00[index]**2+m10[index]**2), m10[index]/sqrt(m00[index]**2+m10[index]**2)])
    clear_output()
    res = []
    for index in trange(len(normalized), desc="Rotating"):
        res.append([north[index] * normalized[index][0] - east[index] * normalized[index][1],
                    north[index] * normalized[index][1] + east[index] * normalized[index][0]])
    clear_output()
    return(pd.DataFrame(res, columns=["Front", "Right"]))
    
#Convert geodetic (latitude, longitude, altitude) data to North,East,Down ECEF frame
def geodetic2ned(lat, lng, alt, lat0=0, lng0=0, alt0=0):
    #if lat0==lng0==alt0==0, assume first row is origin
    if lat0==0 and lng0==0 and alt0==0:
        local = [lat[0], lng[0], alt[0]]
        lng0 = lng[0]
        alt0 = alt[0]
    #Otherwise lat0,lng0, and alt0 are origin
    else:
        local = [lat0, lng0, alt0]
    res = [] #pd.Series(data=None, name="GPS NED")
    for i in trange(len(lat), desc="geodetic2ned"):
        enu = geodetic2enu( lat[i], lng[i], alt[i],
                            local[0], local[1], local[2])
        res.append([enu[1], enu[0], -enu[2]])
    clear_output()
    return(pd.DataFrame(data=res, columns=["North", "East", "Down"]))

#Biases the data by the line of best fit
def linear_bias(ts, signal, times=[], deg=1):
    if type(signal) == list:
        signal = pd.Series(signal)
    if times:
        indices = ts[(ts<times[1]) & (ts>times[0])].index
        fit = Polynomial.fit(ts[indices], signal[indices], deg)
    else:
        fit = Polynomial.fit(ts, signal, deg)
    baseline = [fit(x) for x in ts]
    return(signal-baseline)

#Biases the data by the average calculated at each
#data point
def running_average(ts, signal, times=[], deg=1):
    cumsum = 0
    count = 0
    res = []
    if times:
        for index in trange(len(signal), desc="running_average"):
            if ts[index] > times[0] and ts[index] < times[1]:
                cumsum += signal[index]
                count += 1
                res.append(signal[index] - (cumsum/count))
    else:
        for index in trange(len(signal), desc="running_average"):
            cumsum += signal[index]
            count += 1
            res.append(signal[index] - (cumsum/count))
    clear_output()
    if(type(signal) == pd.core.series.Series):
        return(pd.Series(res, name=signal.name + "_biased"))
    else:
        return(pd.Series(res, name="ys_biased"))

#Matches the faster updating signal to the slower updating one and
#sums the readings between.
def signal_match_and_cumsum(x_ts, x_sig, y_ts, y_sig):
    accumulator = 0
    res = []
    if len(x_ts) >= len(y_ts):
        for j in trange(len(y_ts), desc="signal_match_and_cumsum"):
            for i in range(len(x_ts)):
                if x_ts[i] < y_ts[j]:
                    accumulator += x_sig[i]
                else:
                    res.append(accumulator)
                    accumulator = x_sig[i]
                    break
    else:
        for j in trange(len(x_ts), desc="signal_match_and_cumsum"):
            for i in range(len(y_ts)):
                if y_ts[i] < x_ts[j]:
                    accumulator += y_sig[i]
                else:
                    res.append(accumulator)
                    accumulator = y_sig[i]
                    break
    clear_output()
    return(res)

#length of each row of list of lists
def length(vals, name="length"):
    res = []
    for index in trange(len(vals[0]), desc="length"):
        sqsum = 0
        for val in range(len(vals)):
            sqsum += vals[val][index]**2
        res.append(sqrt(sqsum))
    clear_output()
    return(pd.Series(data=res, name=name)) 
