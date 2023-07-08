from collections import defaultdict

import numpy as np
import pandas as pd

from fastdtw import fastdtw


# 関節の積分を行う
def integral(df: pd.DataFrame, time_column: str = "TimeStamp", is_pos: bool = True,
             is_vec: bool = False) -> pd.DataFrame:
    """
    Parameters:
        df: (位置and回転)座標を含むデータフレーム
        time_column: datetime型のtimestamp
        is_pos: 位置か回転を選択する
        is_vec: 速度か角速度を選択する

    Returns:
        diff: 積分後のデータフレーム
    """

    if is_pos and not is_vec:
        target = "Pos"
        output = "Vec"
    elif is_pos and is_vec:
        target = "Vec"
        output = "Acc"
    elif not is_pos and not is_vec:
        target = "Rot"
        output = "Ang"
    elif not is_pos and is_vec:
        target = "Ang"
        output = "AngAcc"

    if not isinstance(df[time_column].dtype, np.dtypes.DateTime64DType):
        raise TypeError(f"{time_column} is not datetime type.")
    df_copy = df.copy()
    diff = df_copy.loc[:, df_copy.columns.str.contains(target)].diff()
    diff_t = df_copy[time_column].diff().dt.total_seconds()
    diff = diff.div(diff_t, axis=0)
    diff.fillna(0.0, inplace=True)

    diff.columns = diff.columns.str.replace(target, output)
    return pd.concat([df_copy[time_column], diff], axis=1)


# アップサンプリングを行う
def upsampling(df: pd.DataFrame, time_column: str = "TimeStamp", freq: float = 16.6, times: int = 2) -> pd.DataFrame:
    """
    Parameters:
        df: (位置and回転)座標を含むデータフレーム
        time_column: datetime型のtimestamp
        freq: 現在のサンプリング周波数
        times: アップサンプリングを何倍にするか

    Returns:
        df_copy: アップサンプリング後のデータフレーム
    """
    if time_column not in df.columns:
        raise ValueError(f"{time_column} is not in df.")

    df_copy = df.copy()
    start = df_copy[time_column].iloc[0]
    df_copy[time_column] = pd.date_range(start=start, periods=len(df_copy), freq=f"{freq}ms")
    df_copy.set_index(time_column, inplace=True)
    df_copy = df_copy.asfreq(f"{freq / times}ms").interpolate()
    df_copy.reset_index(inplace=True)
    return df_copy


# ノルム計算
def norm(df: pd.DataFrame, is_hand: bool, time_column: str = "TimeStamp") -> pd.DataFrame:
    """
    Args:
        df: x,y,zカラムを含んだデータフレーム
        is_hand: 手のデータかどうか
        time_column:　datetime型のtimestamp

    Returns:
        関節ごとのノルム
    """
    df_copy = df.copy()
    timestamp = df_copy.pop(time_column)
    df_vec_norm = pd.DataFrame()
    for i in range(0, len(df_copy.columns), 3):
        x, y, z = df_copy.iloc[:, i:i + 3]
        if is_hand:
            column = x.split("_")[2]
        else:
            column = x.split("_")[0]
        df_vec_norm[column] = np.linalg.norm([df_copy[x], df_copy[y], df_copy[z]], axis=0)
    return pd.concat([timestamp, df_vec_norm], axis=1)


# dtwの計算
def dtw(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    column1 = df1.max().idxmax()
    column2 = df2.max().idxmax()

    s1 = df1[column1].values
    s2 = df2[column2].values
    _, path = fastdtw(s1, s2)
    return np.array(path), s1, s2


def create_hand_avatar_indexes(path: np.ndarray) -> defaultdict[int, list]:
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
