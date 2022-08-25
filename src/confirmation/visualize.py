import matplotlib.pyplot as plt
import pandas as pd


def simple_time_plot(ts, ys, units=["", ""], title="", atk=0):
    plt.figure(figsize=(12, 8), dpi=80)
    
    if type(ys) == pd.core.frame.DataFrame:
        for col in ys:
            plt.plot(ts, ys[col], label=ys[col].name)
    elif type(ys) == pd.core.frame.Series:
        plt.plot(ts, ys, label=ys.name)
    elif type(ys) == list:
        try:
            #Iterate list of iterables
            for i in range(len(ys)):
                vals = list(iter(ys[i]))
                try:
                    plt.plot(ts, vals, label=ys[i].name)
                except AttributeError:
                    plt.plot(ts, vals, label="ys[" + str(i) + "]")
        except TypeError:
            #Duck Typing, it's a list of non-iterables
            plt.plot(ts, ys, label="ys")
            
    #Point which attack occurs
    if atk != 0:
        plt.axvline(atk, color='red')
    plt.xlabel("Time (" + str(units[0]) + ")")
    plt.ylabel(str(units[1]))
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    