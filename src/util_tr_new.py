import numpy as np
from scipy import interpolate, signal
import pandas as pd
import os.path
import os
import sys
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import List, Dict
from pprint import pprint


# ################################  Library  ########################################
def split_up_down_scans(x_raw, y_raw):
    array_size = len(x_raw)
    is_sweep_going_up = False if (x_raw[0] > x_raw[array_size // 4]) else True
    # print("up to down" if(is_sweep_up_down) else "down to up")
    if is_sweep_going_up:
        return_index = np.argmax(x_raw)
        x_raw_u = x_raw[: return_index + 1]
        x_raw_d = x_raw[return_index:]
        y_raw_u = y_raw[: return_index + 1]
        y_raw_d = y_raw[return_index:]
    else:
        return_index = np.argmin(x_raw)
        x_raw_d = x_raw[: return_index + 1]
        x_raw_u = x_raw[return_index:]
        y_raw_d = y_raw[: return_index + 1]
        y_raw_u = y_raw[return_index:]
    return (x_raw_u, y_raw_u, x_raw_d, y_raw_d)


def get_common_part_of_two_arrays(x1, x2):
    """
    1D-array x1とx2の範囲の共通部分に属する要素のみを残す．
    共通部分の最大値＝それぞれのarrayの最大値の中で最小のもの
    """
    max_of_common_part = np.min([np.max(x1), np.max(x2)])
    min_of_common_part = np.max([np.min(x1), np.min(x2)])
    concatenated_array = np.concatenate([x1, x2])
    common_part = list(set([x for x in concatenated_array if (x <= max_of_common_part and x >= min_of_common_part)]))
    common_part.sort()
    return common_part


def symmetrize(x_raw_u, y_raw_u, x_raw_d=None, y_raw_d=None):
    """1, 4番目の返り値は対称成分．2, 5番目の返り値は反対称化成分，0, 3番目の返り値は共通部分のx座標

    Args:
        x_raw_u (_type_): _description_
        y_raw_u (_type_): _description_
        x_raw_d (_type_, optional): _description_. Defaults to None.
        y_raw_d (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if x_raw_d is None and y_raw_d is None:
        y_int = interpolate.interp1d(x_raw_u, y_raw_u)
        x_ref = np.array([x for x in x_raw_u if (x <= np.max(x_raw_u * -1) and x >= np.min(x_raw_u * -1))])
        y_sym = (y_int(x_ref) + y_int(-1 * x_ref)) / 2
        y_asym = (y_int(x_ref) - y_int(-1 * x_ref)) / 2
        return (x_ref, y_sym, y_asym)
    else:
        try:
            y_int_u = interpolate.interp1d(x_raw_u, y_raw_u)
        except ValueError:
            print("x_raw_u = " + str(x_raw_u))
            print("y_raw_u = " + str(y_raw_u))
            raise ValueError
        try:
            y_int_d = interpolate.interp1d(x_raw_d, y_raw_d)
        except ValueError:
            print("x_raw_d = " + str(x_raw_d))
            print("y_raw_d = " + str(y_raw_d))
            raise ValueError
        x_ref_u = get_common_part_of_two_arrays(x_raw_d, x_raw_u)
        x_ref_d = x_ref_u[::-1]
        y_sym_d = (y_int_d(x_ref_d) + y_int_u(x_ref_u)) / 2
        y_asym_d = (y_int_d(x_ref_d) - y_int_u(x_ref_u)) / 2
        #         y_sym_u = y_sym_d[::-1]
        y_sym_u = y_sym_d
        #         y_asym_u = y_asym_d[::-1]
        y_asym_u = y_asym_d * -1
        return (x_ref_u, y_sym_u, y_asym_u, x_ref_d, y_sym_d, y_asym_d)


def diff_up_down_sweep(B_u, R_u, B_d, R_d):
    """R_uとR_dの差分を取る．トポロジカルホール効果の解析とかに使えるかも

    Args:
        B_u (_type_): _description_
        R_u (_type_): _description_
        B_d (_type_): _description_
        R_d (_type_): _description_

    Returns:
        _type_: _description_
    """
    R_u_int = interpolate.interp1d(B_u, R_u)
    R_d_int = interpolate.interp1d(B_d, R_d)
    B_ref = get_common_part_of_two_arrays(B_d, B_u)
    R_diff = R_u_int(B_ref) - R_d_int(B_ref)
    R_diff_abs = [(R_u_int(x) - R_d_int(x)) if x > 0 else (R_d_int(x) - R_u_int(x)) for x in B_ref]
    return B_ref, R_diff, R_diff_abs


def smooth(x_raw, convolve_size=41):
    """x_rawsの移動平均を取る．畳み込み平均？
    周辺の値を平均することで，ノイズを除去したい時などに

    Args:
        x_raw (_type_):
        convolve_size (int, optional): Defaults to 41.

    Returns:
        _type_: _description_
    """
    # smooth
    convolve_array = np.ones(convolve_size) / convolve_size
    x_raw_convolve = np.convolve(x_raw, convolve_array, mode="valid")
    return x_raw_convolve


def split_arrays(primary_array, secondary_arrays, threshold: float = 1.0, primary_array_name: str = None, seconday_array_names: str = None) -> Dict[float, pd.DataFrame]:
    df = pd.DataFrame(np.array(secondary_arrays).T)
    if seconday_array_names is not None:
        df.columns = seconday_array_names
    df["primary"] = primary_array
    # print(df)
    df["primary_diff"] = df["primary"].diff()
    assert df["primary_diff"].dtype == np.float64, "df['primary_diff'] is not float64. df['primary_diff'].dtype = " + str(df["primary_diff"].dtype)
    assert df["primary"].dtype == np.float64, "df['primary'] is not float64. df['primary'].dtype = " + str(df["primary"].dtype)

    split_dfs = {}
    for _, split_df in df.groupby((df["primary_diff"] > float(threshold)).cumsum()):
        split_df = split_df.drop("primary_diff", axis=1)
        if primary_array_name is not None:
            split_df = split_df.rename(columns={"primary": primary_array_name})
        fixed_primary_value = get_fixed_value_of_split_df(split_df, primary_array_name, round_digit=0)
        split_dfs[fixed_primary_value] = split_df

    return split_dfs


def get_correspond_index(time_stamp, log_time_stamp, time_threshold: float = 0.5):
    print("get_correspond_index")
    log_idxs = []
    for time in time_stamp:
        log_idx = np.argmin(np.abs(log_time_stamp - time))
        log_idxs.append(log_idx)
    return log_idxs


def get_fixed_value_of_split_df(split_df, primary_column, round_digit=0):
    """split_dfのprimary_columnの平均値を取得する．
    同じ温度のデータをのtempの平均値を取得したい時などに
    """
    mean_value = round(split_df[primary_column].mean(), round_digit)
    return mean_value


def normalize_decimal(f):
    """数値fの小数点以下を正規化し、文字列で返す"""

    def _remove_exponent(d):
        return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

    a = Decimal.normalize(Decimal(str(f)))
    b = _remove_exponent(a)
    return str(b)


def save_dfs(dfs: Dict[float, pd.DataFrame], savepath, primary_array_name, pre_string="", post_string=""):
    for fixed_value, df in dfs.items():
        df.to_csv(os.path.join(savepath, pre_string + str(fixed_value) + post_string + ".csv"), index=False)


def make_savepath(filepath, savedir=""):
    # 保存ファイルパスを決定
    filedir = os.path.dirname(filepath)
    filebasename = os.path.basename(filepath)
    filename, fileext = os.path.splitext(filebasename)
    if savedir == "":
        savepath = filedir
    else:
        savepath = os.path.join(filedir, savedir)
        if os.path.isdir(savepath) is False:
            print("makedirs : " + savepath)
            os.makedirs(savepath)
    return filename, fileext, savepath


def get_arrays_from_ppms_df(df, ch_num: List[str]):
    """各々に対応するdat, ch_numに対応する列を取得

    Args:
        df (pd.DataFrame): datファイルを変換したcsvファイルから作成したDataFrame
        ch_num (List[str]): 測定に使用したチャンネル番号．要素数が2の時は縦と横成分，要素数が1の時は縦成分を抽出．横成分は全て0として出力

    Returns:
        _type_: _description_
    """
    time_stamp = df["Time Stamp (sec)"].values
    B = df["Magnetic Field (Oe)"].values * 1e-4
    temp = df["Temperature (K)"].values
    theta = df["Sample Position (deg)"].values.astype(np.float64)
    Rxx = df["Bridge " + str(ch_num[0]) + " Resistivity (Ohm-cm)"].values
    I_Rxx = df["Bridge " + str(ch_num[0]) + " Excitation (uA)"].values * 1e-6
    if len(ch_num) == 2:
        Ryx = df["Bridge " + str(ch_num[1]) + " Resistivity (Ohm-cm)"].values
        I_Ryx = df["Bridge " + str(ch_num[1]) + " Excitation (uA)"].values * 1e-6
    else:
        Ryx = np.full(len(Rxx), 1e-16)
        I_Ryx = np.full(len(I_Rxx), 1e-16)
    assert len(B) == len(temp) == len(theta) == len(Rxx) == len(Ryx) == len(I_Rxx) == len(I_Ryx)
    assert B.dtype == temp.dtype == theta.dtype == Rxx.dtype == Ryx.dtype == I_Rxx.dtype == I_Ryx.dtype == np.float64, "dtype is not float64. dtype = " + str(B.dtype) + ", " + str(temp.dtype) + ", " + str(theta.dtype) + ", " + str(Rxx.dtype) + ", " + str(Ryx.dtype) + ", " + str(I_Rxx.dtype) + ", " + str(I_Ryx.dtype)

    return time_stamp, B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx


def get_arrays_from_log(df, cols: List[str] = []):
    arrays = []
    for col in cols:
        arrays.append(df[col].values)
    return arrays
