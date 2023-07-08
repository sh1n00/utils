from collections import defaultdict

import numpy as np
import pandas as pd


# 関節の積分を行う
def calc_integral(df: pd.DataFrame, time_column: str, is_pos: bool = True, is_vec: bool = False) -> pd.DataFrame:
    """
    Parameters:
    df: (位置and回転)座標を含むデータフレーム
    time_column: datetime型のtimestampのカラム名
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

    diff = df.loc[:, df.columns.str.contains(target)].diff()
    diff_t = df[time_column].diff().dt.total_seconds()
    diff = diff.div(diff_t, axis=0)

    diff.fillna(0.0, inplace=True)
    diff.columns = diff.columns.str.replace(target, output)
    return pd.concat([df[time_column], diff], axis=1)


# アップサンプリングを行う
def calc_upsampling(df: pd.DataFrame, time_column: str, freq: float = 16.6, times: int = 2) -> pd.DataFrame:
    """
    Parameters:
    df: (位置and回転)座標を含むデータフレーム
    time_column: datetime型のtimestampのカラム名
    freq: 現在のサンプリング周波数
    times: アップサンプリングを何倍にするか

    Returns:
    df: アップサンプリング後のデータフレーム
    """
    start = pd.to_datetime(df[time_column], unit="s").iloc[0]
    timestamp = pd.date_range(start=start, periods=len(df), freq=f"{freq}ms")
    df[time_column] = timestamp
    df.set_index(time_column, inplace=True)
    df = df.asfreq(f"{freq//times}ms").interpolate()
    df.reset_index(inplace=True)
    return df


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


def calc_dtw(s1: pd.Series, s2: pd.Series) -> tuple[float, np.ndarray]:
    distance, path = fastdtw(s1, s2)
    return distance, path
