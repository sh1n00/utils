import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import singledispatch
import torch

from utils import settings


@singledispatch
def create_gif(filepath: str, is_show: bool = False) -> None:
    _, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    df = pd.read_csv(f"../data/raw/position/{filename}{ext}")

    timestamp = df.po("time")
    root_pos = df[["joint_Root.x", "joint_Root.y", "joint_Root.z"]]
    df.drop(["joint_Root.x", "joint_Root.y", "joint_Root.z"], axis=1, inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mins, maxs = __get_xyz_min_max(df)

    __plot_setup(ax, *mins, *maxs)

    sc = ax.scatter([], [], [], color="green")

    def update(frame):
        df_i = df.iloc[frame, :]
        dfs = __get_xyz_coordinates(df_i)
        sc._offsets3d = dfs

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        fig.show()


@create_gif.register
def _(df: pd.DataFrame, output_name: str, is_show: bool = False, buffer: int = 20) -> None:
    filename, _ = os.path.splitext(output_name)

    mins, maxs = __get_xyz_min_max(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    __plot_setup(ax, *mins, *maxs)

    sc = ax.scatter([], [], [], color="green")

    def update(frame):
        df_i = df.iloc[frame, :]
        dfs = __get_xyz_coordinates(df_i)
        sc._offsets3d = dfs

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        plt.show()


@create_gif.register
def _(pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mins, maxs = __get_xyz_min_max(pred)

    __plot_setup(ax, *mins, *maxs)

    sc = ax.scatter([], [], [], color="blue")

    def update(frame):
        pred_i = pred[frame]
        preds = __get_xyz_coordinates(pred_i)
        sc._offsets3d = preds

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_pred.gif"), writer="imagemagick")
    if is_show:
        fig.show()


# ラベルデータと予測値を比較する
def create_gif_combine(label: pd.DataFrame, pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mins, maxs = __get_xyz_min_max_combine(label, pred)

    __plot_setup(ax, *mins, *maxs)

    sc1 = ax.scatter([], [], [], label="pred", color="blue")
    sc2 = ax.scatter([], [], [], label="label", color="green")

    def update(frame):
        pred_i = pred[frame]
        label_i = label.iloc[frame, :]

        preds = __get_xyz_coordinates(pred_i)
        labels = __get_xyz_coordinates(label_i)

        sc1._offsets3d = preds
        sc2._offsets3d = labels

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_combine.gif"), writer="imagemagick")
    if is_show:
        fig.show()


# AvatarとHandのPoseを可視化用
def create_gif_combine_hand_avatar(hand: pd.DataFrame, avatar: pd.DataFrame, output_name: str,
                                   is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    mins, maxs = __get_xyz_min_max_combine(hand, avatar)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    __plot_setup(ax, *mins, *maxs)
    sc1 = ax.scatter([], [], [], label="avatar", color="blue")
    sc2 = ax.scatter([], [], [], label="hand", color="green")

    def update(frame):
        avatar_i = avatar.iloc[frame, :]
        hand_i = hand.iloc[frame, :]

        avatars = __get_xyz_coordinates(avatar_i)
        hands = __get_xyz_coordinates(hand_i)

        sc1._offsets3d = avatars
        sc2._offsets3d = hands

    ani = animation.FuncAnimation(fig, update, frames=len(avatar), interval=100)
    os.makedirs(os.path.join(settings.RESULT_DIR, "gif"), exist_ok=True)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_combine.gif"), writer="imagemagick")
    if is_show:
        fig.show()


# 図のセットアップ
def __plot_setup(ax, x_min=-1, y_min=-1, z_min=-1, x_max=1, y_max=1, z_max=1) -> None:
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_zscale("linear")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("x-axis [m]")
    ax.set_ylabel("y-axis [m]")
    ax.set_zlabel("z-axis [m]")

    ax.view_init(elev=-45, azim=90)
    ax.invert_xaxis()


# データのxyzごとに最小値と最大値を算出する
def __get_xyz_min_max(data: pd.DataFrame or torch.Tensor) \
        -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if isinstance(data, pd.DataFrame):
        x_columns = data.columns.str.contains('x')
        y_columns = data.columns.str.contains('y')
        z_columns = data.columns.str.contains('z')
        x_max, x_min = data.loc[:, x_columns].values.max(), data.loc[:, x_columns].values.min()
        y_max, y_min = data.loc[:, y_columns].values.max(), data.loc[:, y_columns].values.min()
        z_max, z_min = data.loc[:, z_columns].values.max(), data.loc[:, z_columns].values.min()
        return (x_min, y_min, z_min), (x_max, y_max, z_max)
    elif isinstance(data, torch.Tensor):
        x = np.array([data[:, i].detach().numpy() for i in range(0, data.shape[1], 3)])
        y = np.array([data[:, i].detach().numpy() for i in range(1, data.shape[1], 3)])
        z = np.array([data[:, i].detach().numpy() for i in range(2, data.shape[1], 3)])
        return (x.min().item(), y.min().item(), z.min().item()), (x.max().item(), y.max().item(), z.max().item())


# 二つのデータの最小値と最大値を取得
def __get_xyz_min_max_combine(data1: pd.DataFrame or torch.Tensor, data2: pd.DataFrame or torch.Tensor) \
        -> tuple[list, list]:
    data1_min, data1_max = __get_xyz_min_max(data1)
    data2_min, data2_max = __get_xyz_min_max(data2)
    mins, maxs = [], []
    for data1_i, data2_i in zip(data1_min, data2_min):
        mins.append(min(data1_i, data2_i))
    for data1_i, data2_i in zip(data1_max, data2_max):
        maxs.append(max(data1_i, data2_i))
    return mins, maxs


# 任意のデータをxyzごとに取得
def __get_xyz_coordinates(data: pd.DataFrame or torch.Tensor) -> tuple[list, list, list]:
    x, y, z = [], [], []
    if isinstance(data, pd.DataFrame):
        for i in range(0, data.shape[0], 3):
            xi, yi, zi = data[i: i + 3]
            x.append(xi)
            y.append(yi)
            z.append(zi)
    elif isinstance(data, torch.Tensor):
        for i in range(0, len(data), 3):
            xi, yi, zi = data[i: i + 3]
            x.append(xi.item())
            y.append(yi.item())
            z.append(zi.item())
    return x, y, z
