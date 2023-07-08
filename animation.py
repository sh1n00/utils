import os

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import singledispatch
import torch

from utils import settings

matplotlib.use('TkAgg')


@singledispatch
def create_gif(filepath: str, is_show: bool = False) -> None:
    _, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    df = pd.read_csv(f"../data/raw/position/{filename}{ext}")

    timestamp = df["time"]
    df.drop("time", axis=1, inplace=True)
    root_pos = df[["joint_Root.x", "joint_Root.y", "joint_Root.z"]]
    df.drop(["joint_Root.x", "joint_Root.y", "joint_Root.z"], axis=1, inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="green")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_zscale("linear")

    x_columns = df.columns.str.contains('x')
    y_columns = df.columns.str.contains('y')
    z_columns = df.columns.str.contains('z')
    x_max, x_min = df.loc[:, x_columns].values.max(), df.loc[:, x_columns].values.min()
    y_max, y_min = df.loc[:, y_columns].values.max(), df.loc[:, y_columns].values.min()
    z_max, z_min = df.loc[:, z_columns].values.max(), df.loc[:, z_columns].values.min()

    ax.set_xlim(x_min - settings.BUFFER, x_max + settings.BUFFER)
    ax.set_ylim(y_min - settings.BUFFER, y_max + settings.BUFFER)
    ax.set_zlim(z_min - settings.BUFFER, z_max + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = df.iloc[frame, :]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi)
            y.append(yi)
            z.append(zi)
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        plt.show()


@create_gif.register
def _(df: pd.DataFrame, output_name: str, is_show: bool = False, buffer: int = 20) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="green")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_zscale("linear")

    x_columns = df.columns.str.contains('x')
    y_columns = df.columns.str.contains('y')
    z_columns = df.columns.str.contains('z')
    x_max, x_min = df.loc[:, x_columns].values.max(), df.loc[:, x_columns].values.min()
    y_max, y_min = df.loc[:, y_columns].values.max(), df.loc[:, y_columns].values.min()
    z_max, z_min = df.loc[:, z_columns].values.max(), df.loc[:, z_columns].values.min()

    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = df.iloc[frame, :]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi)
            y.append(yi)
            z.append(zi)
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        plt.show()


@create_gif.register
def _(pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="blue")

    x = np.array([pred[:, i].detach().numpy() for i in range(0, pred.shape[1], 3)])
    y = np.array([pred[:, i].detach().numpy() for i in range(1, pred.shape[1], 3)])
    z = np.array([pred[:, i].detach().numpy() for i in range(2, pred.shape[1], 3)])
    ax.set_xlim(x.min() - settings.BUFFER, x.max() + settings.BUFFER)
    ax.set_ylim(y.min() - settings.BUFFER, y.max() + settings.BUFFER)
    ax.set_zlim(z.min() - settings.BUFFER, z.max() + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = pred[frame]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi.item())
            y.append(yi.item())
            z.append(zi.item())
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_pred.gif"), writer="imagemagick")
    if is_show:
        fig.show()


def create_gif_combine(label: pd.DataFrame, pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter([], [], [], label="pred", color="blue")
    sc2 = ax.scatter([], [], [], label="label", color="green")

    x = np.array([pred[:, i].detach().numpy() for i in range(0, pred.shape[1], 3)])
    y = np.array([pred[:, i].detach().numpy() for i in range(1, pred.shape[1], 3)])
    z = np.array([pred[:, i].detach().numpy() for i in range(2, pred.shape[1], 3)])

    x_columns = label.columns.str.contains('x')
    y_columns = label.columns.str.contains('y')
    z_columns = label.columns.str.contains('z')
    x_max, x_min = label.loc[:, x_columns].values.max(), label.loc[:, x_columns].values.min()
    y_max, y_min = label.loc[:, y_columns].values.max(), label.loc[:, y_columns].values.min()
    z_max, z_min = label.loc[:, z_columns].values.max(), label.loc[:, z_columns].values.min()

    ax.set_xlim(min(x.min(), x_min) - settings.BUFFER, max(x.max(), x_max) + settings.BUFFER)
    ax.set_ylim(min(y.min(), y_min) - settings.BUFFER, max(y.max(), y_max) + settings.BUFFER)
    ax.set_zlim(min(z.min(), z_min) - settings.BUFFER, max(z.max(), z_max) + settings.BUFFER)

    ax.set_xlabel("x-axis [m]")
    ax.set_ylabel("y-axis [m]")
    ax.set_zlabel("z-axis [m]")

    ax.legend()

    def update(frame):
        df_pred = pred[frame]
        df_label = label.iloc[frame, :]
        x_pred, y_pred, z_pred = [], [], []
        x_label, y_label, z_label = [], [], []
        for index in range(0, len(df_label), 3):
            xi_p, yi_p, zi_p = df_pred[index:index + 3]
            xi_l, yi_l, zi_l = df_label[index:index + 3]
            x_pred.append(xi_p.item())
            y_pred.append(yi_p.item())
            z_pred.append(zi_p.item())
            x_label.append(xi_l)
            y_label.append(yi_l)
            z_label.append(zi_l)
        sc1._offsets3d = (x_pred, y_pred, z_pred)
        sc2._offsets3d = (x_label, y_label, z_label)

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_combine.gif"), writer="imagemagick")
    if is_show:
        fig.show()


# AvatarとHandのPoseを可視化用
def create_gif_combine_hand_avatar(hand: pd.DataFrame, avatar: pd.DataFrame, output_name: str, is_show: bool = False,
                                   buffer: int = 0) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter([], [], [], label="avatar", color="blue")
    sc2 = ax.scatter([], [], [], label="hand", color="green")

    x = avatar.columns.str.contains('x')
    y = avatar.columns.str.contains('y')
    z = avatar.columns.str.contains('z')
    x_amax, x_amin = avatar.loc[:, x].values.max(), avatar.loc[:, x].values.min()
    y_amax, y_amin = avatar.loc[:, y].values.max(), avatar.loc[:, y].values.min()
    z_amax, z_amin = avatar.loc[:, z].values.max(), avatar.loc[:, z].values.min()

    x_columns = hand.columns.str.contains('x')
    y_columns = hand.columns.str.contains('y')
    z_columns = hand.columns.str.contains('z')
    x_max, x_min = hand.loc[:, x_columns].values.max(), hand.loc[:, x_columns].values.min()
    y_max, y_min = hand.loc[:, y_columns].values.max(), hand.loc[:, y_columns].values.min()
    z_max, z_min = hand.loc[:, z_columns].values.max(), hand.loc[:, z_columns].values.min()

    ax.set_xlim(min(x_amin, x_min) - buffer, max(x_amax, x_max) + buffer)
    ax.set_ylim(min(y_amin, y_min), max(y_amax, y_max))
    ax.set_zlim(min(z_amin, z_min), max(z_amax, z_max))

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    ax.view_init(elev=-45, azim=90)

    ax.legend()

    def update(frame):
        df_avatar = avatar.iloc[frame, :]
        df_hand = hand.iloc[frame, :]
        x_pred, y_pred, z_pred = [], [], []
        x_label, y_label, z_label = [], [], []
        for index in range(0, len(df_hand), 3):
            xi_p, yi_p, zi_p = df_avatar[index:index + 3]
            xi_l, yi_l, zi_l = df_hand[index:index + 3]
            x_pred.append(xi_p)
            y_pred.append(yi_p)
            z_pred.append(zi_p)
            x_label.append(xi_l)
            y_label.append(yi_l)
            z_label.append(zi_l)
        sc1._offsets3d = (x_pred, y_pred, z_pred)
        sc2._offsets3d = (x_label, y_label, z_label)

    ax.invert_xaxis()
    ani = animation.FuncAnimation(fig, update, frames=len(avatar), interval=100)
    os.makedirs(os.path.join(settings.RESULT_DIR, "gif"), exist_ok=True)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_combine.gif"), writer="imagemagick")
    if is_show:
        plt.show()
