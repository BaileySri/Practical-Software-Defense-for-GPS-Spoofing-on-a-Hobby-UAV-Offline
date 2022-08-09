import matplotlib.pyplot as plt

def simple_time_plot(ts, ys):
    plt.plot(ts, ys)
    plt.xlabel(ts.name)
    plt.ylabel(ys.name)
    plt.show()
    