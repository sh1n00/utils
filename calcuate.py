from collections import defaultdict

import numpy as np
import pandas as pd

from utils import myenum


# 関節の速度を計算する
def calc_velocity(df: pd.DataFrame) -> pd.DataFrame:
    diff_pos = df[myenum.DefaultColumns.get_all_default_columns()].diff()
    diff_t = df["time"].diff()
    df_velocity = diff_pos.div(diff_t, axis=0)

    # 最初の速度は0
    df_velocity.fillna(0.0, inplace=True)
    df_velocity.columns = myenum.VelocityColumns.get_all_velocities()
    return pd.concat([df, df_velocity], axis=1)


def calc_velocity_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    diff_pos = df[columns].diff()
    diff_t = df["time"].diff()
    df_velocity = diff_pos.div(diff_t, axis=0)

    # 最初の速度は0
    df_velocity.fillna(0.0, inplace=True)
    df_velocity.columns = myenum.VelocityColumns.get_all_velocities()
    return pd.concat([df, df_velocity], axis=1)


# 関節の加速度を計算する
def calc_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    diff_vec = df[myenum.VelocityColumns.get_all_velocities()].diff()
    diff_t = df["time"].diff()
    df_acceleration = diff_vec.div(diff_t, axis=0)

    # 最初の加速度は0
    df_acceleration.fillna(0.0, inplace=True)
    df_acceleration.columns = myenum.AccelerationColumns.get_all_accelerations()
    return pd.concat([df, df_acceleration], axis=1)


# 関節の角速度を計算する
def calc_angular_velocity(df: pd.DataFrame) -> pd.DataFrame:
    diff_vec = df[myenum.DefaultColumns.get_all_default_columns()].diff()
    diff_t = df["time"].diff()
    df_acceleration = diff_vec.div(diff_t, axis=0)

    # 最初の加速度は0
    df_acceleration.fillna(0.0, inplace=True)
    df_acceleration.columns = myenum.AngularVelocityColumns.get_all_angular_velocities()
    return pd.concat([df, df_acceleration], axis=1)


# 関節の角加速度を計算する
def calc_angular_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    diff_vec = df[myenum.AngularVelocityColumns.get_all_angular_velocities()].diff()
    diff_t = df["time"].diff()
    df_acceleration = diff_vec.div(diff_t, axis=0)

    # 最初の加速度は0
    df_acceleration.fillna(0.0, inplace=True)
    df_acceleration.columns = myenum.AngularAccelerationColumns.get_all_angular_accelerations()
    return pd.concat([df, df_acceleration], axis=1)


def create_hand_avatar_indexes(path: np.ndarray) -> defaultdict:
    d = defaultdict(lambda: list())
    for k, v in path:
        d[k].append(v)
    return d


def create_avatar_fixed_df(avatar_record: pd.DataFrame, d: defaultdict) -> pd.DataFrame:
    record = pd.DataFrame()
    if "TimeStamp" in avatar_record.columns:
        avatar_record = avatar_record.drop("TimeStamp", axis=1)
    for k, v in d.items():
        record = pd.concat([record, pd.DataFrame(avatar_record.iloc[v, :].mean(axis=0)).T], axis=0)
    return record


if __name__ == "__main__":
    df = calc_velocity(pd.read_csv(f"../position/dataset-1_walk-back_angry_001_pos.csv"))
