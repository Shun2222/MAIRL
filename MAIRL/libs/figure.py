import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
import datetime
import os
import numpy as np
from MAIRL.environment import *
from MAIRL.libs.traj_util import *

def make_path(save_dir="./", file_name="Non", extension=".png"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name == "Non":
        file_name = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')) 
    file_name += extension
    return os.path.join(save_dir, file_name)

def plot_on_grid(values, state_size, file_name="Non", folder="./", set_annot=True, save=True, show=False, title=""):
    values = np.array(values)
    if len(values.shape) < 2:
        values = values.reshape(state_size)
    plt.figure()
    plt.title(title)
    img = sns.heatmap(values,annot=set_annot,square=True,cmap='PuRd')
    if save:
        file_path = make_path(folder, file_name)
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()
    return img

def arrow_plot(data, actions, state_size, file_name="Non", folder="./", save=True, show=False, title=""):
    data = np.array(data)
    if len(data.shape) < 2:
        data = data.reshape(state_size)
    actions = np.array(actions)
    if len(actions.shape) < 2:
        actions = actions.reshape(state_size)
    e=[['0']*state_size[1] for _ in range(state_size[0])]
    e[0][0] = 'S'
    e[state_size[0]-1][state_size[1]-1]='G'
    env = GridWorldEnv(grid=e)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
    sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax, annot=True, square=True)
    ax.set_title(title)
    for s in range(state_size[0]*state_size[1]):
         y, x = divmod(s, state_size[1])
         if actions[y][x]==-1:
            continue
         a = np.array(env.action_to_vec(actions[y][x]))
         ax.arrow(x+0.5, state_size[0]-1.5+y, a[0]*0.6, a[1]*0.6, color='blue', head_width=0.2, head_length=0.2)
    file_path = make_path(folder, file_name)
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()

def trajs_plot(trajs, state_size, file_name="Non", folder="./", save=True, show=False, title=""):
    action_vecs = [[] for _ in range(len(trajs))]
    for i in range(len(trajs)):
        action_vecs[i]  = traj_to_action_vecs(trajs[i], state_size)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [8,8])
    ax.set_title(title)
    ax.grid(True)
    ax.set_xticks(np.arange(state_size[1]+1))
    ax.set_yticks(np.arange(state_size[0]+1))
    ax.set_xlim(0, state_size[1])
    ax.set_ylim(0, state_size[0])
    cmap = plt.cm.jet
    cNorm  = colors.Normalize(vmin=0, vmax=len(trajs))
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
    for i in range(len(trajs)):
        colorVal = scalarMap.to_rgba(i)
        y, x = divmod(trajs[i][0], state_size[1])
        c = patches.Circle( xy=(x+0.5+0.35, state_size[0]-1-y+0.5+0.35), radius=0.1, color=colorVal)
        ax.text(x+0.5+0.25, state_size[0]-1-y+0.5+0.25, str(i), size=8)
        ax.add_patch(c)
        for s in range(state_size[0]*state_size[1]):
            if all(action_vecs[i][s]==[0,0]):
                continue
            a = action_vecs[i][s]
            y, x = divmod(s, state_size[1])
            ax.arrow(x+0.5, state_size[0]-1-y+0.5, a[1]*0.3, -a[0]*0.3, color=colorVal, head_width=0.2, head_length=0.2)
    file_path = make_path(folder, file_name)
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()

class make_gif():
    def __init__(self):
        self.datas = []

    def add_data(self, d):
        self.datas += [d]

    def add_datas(self, ds):
        self.datas += ds

    def make(self, state_size, folder="./", file_name="Non", save=True, show=False):
        def make_heatmap(i):
            ax.cla()
            ax.set_title("Iteration="+str(i))
            data = np.array(self.datas[i])
            if len(data.shape) < 2:
                data = data.reshape(state_size)
            sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax)
        fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
        ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
        if save:
            file_path = make_path(folder, file_name, extension=".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()

    def reset(self):
        plt.close()
        self.datas = []

    def make_test(self, state_size, folder="./", file_name="Non", save=True, show=False):
        def make_arrow(i, state_size):
            ax.cla()
            ax.set_title("Iteration="+str(i))
            trajs = self.datas[i]
            for i in range(len(trajs)):
                action_vecs[i]  = traj_to_action_vecs(trajs[i], state_size)
            ax.grid(True)
            ax.set_xticks(np.arange(state_size[1]+1))
            ax.set_yticks(np.arange(state_size[0]+1))
            ax.set_xlim(0, state_size[1])
            ax.set_ylim(0, state_size[0])
            for i in range(len(trajs)):
                colorVal = scalarMap.to_rgba(i)
                for s in range(state_size[0]*state_size[1]):
                    if all(action_vecs[i][s]==[0,0]):
                        continue
                    a = action_vecs[i][s]
                    y, x = divmod(s, state_size[1])
                    ax.arrow(x+0.5, state_size[0]-1-y+0.5, a[1]*0.3, -a[0]*0.3, color=colorVal, head_width=0.2, head_length=0.2)

        fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [8,8])
        cmap = plt.cm.jet
        cNorm  = colors.Normalize(vmin=0, vmax=fms)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
        ani = animation.FuncAnimation(fig=fig, func=make_arrow, fargs=(state_size), frames=fms, interval=500, blit=False)
        if save:
            file_path = make_path(folder, file_name, extension=".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()        
    
"""
from MAIRL.environment import *
from MAIRL.libs.figure import *
trajs_plot([[0,1,2],[8,5,4,3,0]], [3,3], show=True)
"""