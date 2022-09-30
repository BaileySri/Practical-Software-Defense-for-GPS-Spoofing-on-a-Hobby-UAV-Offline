import matplotlib.pyplot as plt
import pandas as pd


def simple_time_plot(ts, ys, units=["", ""], title="", atk=0, lines=[]):
    if(len(ts) != len(ys)):
        print("Mismatched timestamp and signal lengths: %s Timestamps, %s Signals" % (str(len(ts)), str(len(ys))))
    else:
        plt.figure(figsize=(12, 8), dpi=80)
        if type(ys) == pd.core.frame.DataFrame:
            for col in range(len(ys.columns)):
                plt.plot(ts[col], ys.iloc[:,col], label=ys.iloc[:,col].name)
        elif type(ys) == pd.core.frame.Series:
            plt.plot(ts, ys, label=ys.name)
        elif type(ys) == list:
            if (type(ys[0]) != pd.core.frame.Series) and (type(ys[0]) != list):
                plt.plot(ts, ys, label="ys")
            else:
                try:
                    #Iterate list of iterables
                    for i in range(len(ys)):
                        vals = list(iter(ys[i]))
                        try:
                            plt.plot(ts[i], vals, label=ys[i].name)
                        except AttributeError:
                            plt.plot(ts[i], vals, label="ys[" + str(i) + "]")
                except TypeError:
                    #Duck Typing, it's a list of non-iterables
                    plt.plot(ts[i], ys, label="ys")

        #Point which attack occurs
        if atk != 0:
            plt.axvline(atk, color='r')
        #Other lines
        if lines:
            for line in lines:
                plt.axvline(line)
        plt.xlabel("Time (" + str(units[0]) + ")")
        plt.ylabel(str(units[1]))
        if title:
            plt.title(title)
        plt.legend()
        plt.show()
    