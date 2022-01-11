import global_var
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

def g(t):
    return np.sin(np.pi*t)

def perform_graph(time_perform, trans_perform, dis_perform):
    generation = global_var.n_generations
    num = global_var.task_num
    x_ = [["r_time", "b_time"], ["r_trans", "b_trans"], ["r_distance", "b_distance"]]
    d = len(num) * 100 + 3 * 10

    for g in range(len(num)):
        for i in range(3):
            t = time_perform[g]
            trans = trans_perform[g]
            dis = dis_perform[g]

            data = [t, trans, dis]

            d += 1
            plt.subplot(d)
            plt.bar(x_[i], data[i], width=0.3)
            text(x_[i], data[i])

    plt.show()

def text(x, y):
    for i, v in enumerate(y):
        plt.text(x[i], v, v, fontsize=9, color='#ff6600')

if __name__=='__main__':
    perform_graph([[123,321]], [[24,8]], [[30000,20000]])