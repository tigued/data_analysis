import numpy as np
from scipy import interpolate, signal
import pandas as pd
import os.path, os, sys
import matplotlib.pyplot as plt
from decimal import Decimal

#################################  Library  ########################################
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
    # 1D-array x1とx2の範囲の共通部分に属する要素のみを残す
    # 共通部分の最大値＝それぞれのarrayの最大値の中で最小のもの
    max_of_common_part = np.min([np.max(x1), np.max(x2)])
    min_of_common_part = np.max([np.min(x1), np.min(x2)])
    concatenated_array = np.concatenate([x1, x2])
    common_part = list(set([x for x in concatenated_array if (x <= max_of_common_part and x >= min_of_common_part)]))
    common_part.sort()
    return common_part


def symmetrize(x_raw_u, y_raw_u, x_raw_d=None, y_raw_d=None):
    if x_raw_d is None and y_raw_d is None:
        y_int = interpolate.interp1d(x_raw_u, y_raw_u)
        x_ref = np.array([x for x in x_raw_u if (x <= np.max(x_raw_u * -1) and x >= np.min(x_raw_u * -1))])
        y_sym = (y_int(x_ref) + y_int(-1 * x_ref)) / 2
        y_asym = (y_int(x_ref) - y_int(-1 * x_ref)) / 2
        return (x_ref, y_sym, y_asym)
    else:
        y_int_u = interpolate.interp1d(x_raw_u, y_raw_u)
        y_int_d = interpolate.interp1d(x_raw_d, y_raw_d)
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
    R_u_int = interpolate.interp1d(B_u, R_u)
    R_d_int = interpolate.interp1d(B_d, R_d)
    B_ref = get_common_part_of_two_arrays(B_d, B_u)
    R_diff = R_u_int(B_ref) - R_d_int(B_ref)
    R_diff_abs = [(R_u_int(x) - R_d_int(x)) if x > 0 else (R_d_int(x) - R_u_int(x)) for x in B_ref]
    return B_ref, R_diff, R_diff_abs


def smooth(x_raw, convolve_size=41):
    # smooth
    convolve_array = np.ones(convolve_size) / convolve_size
    x_raw_convolve = np.convolve(x_raw, convolve_array, mode="valid")
    return x_raw_convolve


def analyze_Hall(B_raw, Rxx_raw, Ryx_raw, fixed_temp=0, thickness=0):
    # 温度固定、磁場スイープ
    # 折返し有り無し判定(磁性体か非磁性か)
    is_sweep_going_up = True if B_raw[0] * B_raw[len(B_raw) - 1] > 0 else False
    if is_sweep_going_up:
        # 往復あり(磁性体)
        B_ref_u, Rxx_u, _, B_ref_d, Rxx_d, _ = symmetrize(*split_up_down_scans(B_raw, Rxx_raw))
        B_ref_u, _, Ryx_u, B_ref_d, _, Ryx_d = symmetrize(*split_up_down_scans(B_raw, Ryx_raw))
        Gxx_d = Rxx_d / (Rxx_d**2 + Ryx_d**2)
        Gxy_d = Ryx_d / (Rxx_d**2 + Ryx_d**2)
        Gxx_u = Rxx_u / (Rxx_u**2 + Ryx_u**2)
        Gxy_u = Ryx_u / (Rxx_u**2 + Ryx_u**2)
        Rxx_d_int = interpolate.interp1d(B_ref_d, Rxx_d)
        Ryx_d_int = interpolate.interp1d(B_ref_d, Ryx_d)
        Ryx_u_int = interpolate.interp1d(B_ref_u, Ryx_u)
        Ryx_d_sq = Ryx_d**2
        B_over_Ryx = B_ref_d / Ryx_d
        temp = np.full_like(B_ref_u, fixed_temp)
        RyxA = (Ryx_d_int(0) - Ryx_u_int(0)) / 2
        Rxx0 = Rxx_d_int(0)
        HallAngle_d = Ryx_d / Rxx_d
        HallAngle_u = Ryx_u / Rxx_u
        HallAngle0 = RyxA / Rxx0
        # down scanのB>0だけ取り出してフィッティングする
        subdf_d_pos = pd.DataFrame(np.array([B_raw, Ryx_raw]).T, columns=["B", "Ryx"]).query("B > 0")
        B_ref_d_pos = subdf_d_pos["B"].values
        Ryx_d_pos = subdf_d_pos["Ryx"].values
        fit = np.polyfit(B_ref_d_pos, Ryx_d_pos, 1)
        carrier2D = 1e-4 / (1.602e-19 * fit[0])
        mobility = 1e4 * fit[0] / Rxx0
        if thickness == 0:
            # 2D
            data = np.array(
                [
                    temp,
                    B_ref_d,
                    B_ref_u,
                    Rxx_d,
                    Rxx_u,
                    Ryx_d,
                    Ryx_u,
                    Gxx_d,
                    Gxx_u,
                    Gxy_d,
                    Gxy_u,
                    Ryx_d_sq,
                    B_over_Ryx,
                    HallAngle_d,
                    HallAngle_u,
                ]
            )
            columns = [
                "temp",
                "B_ref_d",
                "B_ref_u",
                "Rxx_d",
                "Rxx_u",
                "Ryx_d",
                "Ryx_u",
                "Gxx_d",
                "Gxx_u",
                "Gxy_d",
                "Gxy_u",
                "Ryx_d_sq",
                "B_over_Ryx",
                "HallAngle_d",
                "HallAngle_u",
            ]
            summary = pd.Series(
                [fixed_temp, Rxx0, RyxA, RyxA, carrier2D, mobility, np.abs(carrier2D), np.abs(mobility), HallAngle0],
                index=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                ],
            )
        else:
            # 3D
            Rxx_u_3D = Rxx_u * thickness * 1e2
            Rxx_d_3D = Rxx_d * thickness * 1e2
            Ryx_u_3D = Ryx_u * thickness * 1e2
            Ryx_d_3D = Ryx_d * thickness * 1e2
            Gxx_d_3D = Gxx_d / (thickness * 1e2)
            Gxy_d_3D = Gxy_d / (thickness * 1e2)
            Gxx_u_3D = Gxx_u / (thickness * 1e2)
            Gxy_u_3D = Gxy_u / (thickness * 1e2)
            RyxA_3D = RyxA * thickness * 1e2
            Rxx0_3D = Rxx0 * thickness * 1e2
            carrier3D = carrier2D / (thickness * 1e2)
            data = np.array(
                [
                    temp,
                    B_ref_d,
                    B_ref_u,
                    Rxx_d,
                    Rxx_u,
                    Ryx_d,
                    Ryx_u,
                    Gxx_d,
                    Gxx_u,
                    Gxy_d,
                    Gxy_u,
                    Ryx_d_sq,
                    B_over_Ryx,
                    HallAngle_d,
                    HallAngle_u,
                    Rxx_u_3D,
                    Rxx_d_3D,
                    Ryx_u_3D,
                    Ryx_d_3D,
                    Gxx_u_3D,
                    Gxx_d_3D,
                    Gxy_u_3D,
                    Gxy_d_3D,
                ]
            )
            columns = [
                "temp",
                "B_ref_d",
                "B_ref_u",
                "Rxx_d",
                "Rxx_u",
                "Ryx_d",
                "Ryx_u",
                "Gxx_d",
                "Gxx_u",
                "Gxy_d",
                "Gxy_u",
                "Ryx_d_sq",
                "B_over_Ryx",
                "HallAngle_d",
                "HallAngle_u",
                "Rxx_u_3D",
                "Rxx_d_3D",
                "Ryx_u_3D",
                "Ryx_d_3D",
                "Gxx_u_3D",
                "Gxx_d_3D",
                "Gxy_u_3D",
                "Gxy_d_3D",
            ]
            summary = pd.Series(
                [
                    fixed_temp,
                    Rxx0,
                    RyxA,
                    RyxA,
                    carrier2D,
                    mobility,
                    np.abs(carrier2D),
                    np.abs(mobility),
                    HallAngle0,
                    Rxx0_3D,
                    RyxA_3D,
                    carrier3D,
                    np.abs(carrier3D),
                ],
                index=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                    "Rxx0T_3D",
                    "RyxA_3D",
                    "carrier3D",
                    "carrier3D_abs",
                ],
            )
    else:
        # 往復なし(非磁性体)
        B_ref, Rxx, _ = symmetrize(B_raw, Rxx_raw)
        B_ref, _, Ryx = symmetrize(B_raw, Ryx_raw)
        Gxx = Rxx / (Rxx**2 + Ryx**2)
        Gxy = Ryx / (Rxx**2 + Ryx**2)
        Rxx_int = interpolate.interp1d(B_ref, Rxx)
        Ryx_int = interpolate.interp1d(B_ref, Ryx)
        temp = np.full_like(B_ref, fixed_temp)
        Rxx0 = Rxx_int(0)
        Ryx0 = Ryx_int(0)
        HallAngle = Ryx / Rxx
        try:
            fit = np.polyfit(B_ref, Ryx, 1)
        except:
            print(str(fixed_temp) + "K")
            import traceback

            traceback.print_exc()
        carrier2D = 1e-4 / (1.602e-19 * fit[0])
        mobility = 1e4 * fit[0] / Rxx0
        if thickness == 0:
            # 2D
            data = np.array([temp, B_ref, Rxx, Ryx, Gxx, Gxy, HallAngle, np.abs(HallAngle)])
            columns = ["temp", "B_ref", "Rxx", "Ryx", "Gxx", "Gxy", "HallAngle", "HallAngle_abs"]
            summary = pd.Series(
                [fixed_temp, carrier2D, mobility, np.abs(carrier2D), np.abs(mobility)],
                index=["temps", "carrier2D", "mobility", "carrier2D_abs", "mobility_abs"],
            )
        else:
            # 3D
            Rxx_3D = Rxx * thickness * 1e2
            Ryx_3D = Ryx * thickness * 1e2
            Gxx_3D = Gxx / (thickness * 1e2)
            Gxy_3D = Gxy / (thickness * 1e2)
            Rxx0_3D = Rxx0 * thickness * 1e2
            carrier3D = carrier2D / (thickness * 1e2)
            data = np.array([temp, B_ref, Rxx, Ryx, Gxx, Gxy, HallAngle, np.abs(HallAngle), Rxx_3D, Ryx_3D, Gxx_3D, Gxy_3D])
            columns = [
                "temp",
                "B_ref",
                "Rxx",
                "Ryx",
                "Gxx",
                "Gxy",
                "HallAngle",
                "HallAngle_abs",
                "Rxx_3D",
                "Ryx_3D",
                "Gxx_3D",
                "Gxy_3D",
            ]
            summary = pd.Series(
                [fixed_temp, carrier2D, mobility, np.abs(carrier2D), np.abs(mobility), carrier3D, np.abs(carrier3D)],
                index=["temps", "carrier2D", "mobility", "carrier2D_abs", "mobility_abs", "carrier3D", "carrier3D_abs"],
            )
    analyzed_df = pd.DataFrame(data=data.T, columns=columns)
    return analyzed_df, summary

def split_arrays(primary_array, secondary_arrays, threshold = 1, primary_array_name = None, seconday_array_names = None):
    df = pd.DataFrame(np.array(secondary_arrays).T)
    if (seconday_array_names is not None):
          df.columns = seconday_array_names
    df["primary"] = primary_array
    # print(df)
    df["primary_diff"] = df["primary"].diff()
    split_dfs = []
    for _, split_df in df.groupby((df["primary_diff"] > threshold).cumsum()):
        split_df = split_df.drop("primary_diff", axis = 1)
        if (primary_array_name is not None):
            split_df = split_df.rename(columns = {"primary" : primary_array_name})
        split_dfs.append(split_df)
    return split_dfs

def get_fixed_value_of_split_df(split_df, primary_column, round_digit = 1):
    mean_value = round(split_df[primary_column].mean(), round_digit)
    return mean_value

def normalize_decimal(f):
    """数値fの小数点以下を正規化し、文字列で返す"""
    def _remove_exponent(d):
        return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()
    a = Decimal.normalize(Decimal(str(f)))
    b = _remove_exponent(a)
    return str(b)

def save_dfs(dfs, savepath, primary_array_name, pre_string = "", post_string = ""):
    for df in dfs:
        fixed_value = get_fixed_value_of_split_df(df, primary_array_name, 1)
        df.to_csv(os.path.join(savepath, pre_string + normalize_decimal(fixed_value) + post_string + ".csv"), index=False)

def do_Hall_analysis_for_dfs(dfs, thickness = 0):
    analyzed_dfs = [] # 各温度における解析済みデータ
    summaries = pd.DataFrame()  # 各温度点の特徴量をまとめたデータ(indexは温度点)
    for df in dfs:
        temp = df["temp"].values
        B = df["B"].values
        Rxx = df["Rxx"].values
        Ryx = df["Ryx"].values
        fixed_temp = get_fixed_value_of_split_df(df, "temp", round_digit = 1)
        # Hall解析
        analyzed_df, summary = analyze_Hall(B, Rxx, Ryx, fixed_temp, thickness)
        analyzed_dfs.append(analyzed_df)
        # summaryデータのdfを更新
        summaries = pd.concat([summaries, summary], axis = 1)
    #summariesを転置
    summaries = summaries.T.reset_index(drop=True)
    # summaryデータの処理
    # すべてがNaNな行、列を削除
    summaries = summaries.dropna(how="all").dropna(how="all", axis=1)
    # RyxA_normの規格化
    if "RyxA_norm" in summaries.columns:
        summaries["RyxA_norm"] = summaries["RyxA_norm"].map(lambda x: x / summaries["RyxA_norm"][0])
    return analyzed_dfs, summaries

####################### plot ###########################
def plot_B_dep_data(analyzed_dfs, raw_dfs, sample_str=""):
    # グラフ描画準備
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    fig.text(0.1, 0.9, sample_str)

    for df in raw_dfs:
        # プロット:raw
        ax[0][1].plot(df["B"], df["Rxx"])
        ax[1][1].plot(df["B"], df["Ryx"])
        ax[0][1].set_ylabel("Rxx_raw (Ohm)")
        ax[1][1].set_ylabel("Ryx_raw (Ohm)")
        ax[1][1].set_xlabel("B (T)")

        ax[0][2].plot(df["B"], df["I_Rxx"])
        ax[1][2].plot(df["B"], df["I_Ryx"])
        ax[0][2].set_ylabel("I_Rxx (A)")
        ax[1][2].set_ylabel("I_Ryx (A)")
        ax[1][2].set_xlabel("B (T)")

    for df in analyzed_dfs:
        # プロット:解析データ
        if "B_ref_d" in df.columns:
            # AHE
            ax[0][0].plot(pd.concat([df["B_ref_d"], df["B_ref_u"]]), pd.concat([df["Rxx_d"], df["Rxx_u"]]))
            ax[1][0].plot(pd.concat([df["B_ref_d"], df["B_ref_u"]]), pd.concat([df["Ryx_d"], df["Ryx_u"]]))
        else:
            # OHE
            ax[0][0].plot(df["B_ref"], df["Rxx"])
            ax[1][0].plot(df["B_ref"], df["Ryx"])
        ax[0][0].set_ylabel("Rxx (Ohm)")
        ax[1][0].set_ylabel("Ryx (Ohm)")
        ax[1][0].set_xlabel("B (T)")
    return fig


def plot_summary_data(summaries, sample_str=""):
    # グラフ描画準備
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.3)
    fig.text(0.1, 0.9, sample_str)
    # プロット:temp-dep
    ax[0].plot(summaries["temps"], summaries["carrier2D_abs"], marker="o")
    ax[0].set_ylabel("carrier 2D(cm-2)")
    if "carrier3D_abs" in summaries.columns:
        ax[1].plot(summaries["temps"], summaries["carrier3D_abs"], marker="o")
        ax[1].set_ylabel("carrier 3D(cm-3)")
    if "RyxA" in summaries.columns:
        ax[2].plot(summaries["temps"], summaries["RyxA"], marker="o")
        ax[2].set_ylabel("RyxA (Ohm)")
    else:
        ax[2].plot(summaries["temps"], summaries["mobility"], marker="o")
        ax[2].set_ylabel("mobility (cm2V-1s-1)")
    ax[2].set_xlabel("T (K)")
    return fig


def plot_RT_data(df_RT, sample_str=""):
    if "Rxx_3D" in df_RT.columns:
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 9))
        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        fig.text(0.1, 0.9, sample_str)
        ax[0][0].plot(df_RT["temp"], df_RT["Rxx"])
        ax[0][1].plot(df_RT["temp"], df_RT["Ryx"])
        ax[0][0].set_ylabel("Rxx_2D (Ohm)")
        ax[0][1].set_ylabel("Ryx_2D (Ohm)")
        ax[1][0].plot(df_RT["temp"], df_RT["Rxx_3D"])
        ax[1][1].plot(df_RT["temp"], df_RT["Ryx_3D"])
        ax[1][0].set_ylabel("Rxx_3D (Ohm cm)")
        ax[1][1].set_ylabel("Ryx_3D (Ohm cm)")
        ax[2][0].plot(df_RT["temp"], df_RT["I_Rxx"])
        ax[2][1].plot(df_RT["temp"], df_RT["I_Ryx"])
        ax[2][0].set_ylabel("I_Rxx (A)")
        ax[2][1].set_ylabel("I_Ryx (A)")
        ax[2][0].set_xlabel("T (K)")
        ax[2][1].set_xlabel("T (K)")
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        fig.text(0.1, 0.9, sample_str)
        ax[0][0].plot(df_RT["temp"], df_RT["Rxx"])
        ax[0][1].plot(df_RT["temp"], df_RT["Ryx"])
        ax[0][0].set_ylabel("Rxx_2D (Ohm)")
        ax[0][1].set_ylabel("Ryx_2D (Ohm)")
        ax[1][0].plot(df_RT["temp"], df_RT["I_Rxx"])
        ax[1][1].plot(df_RT["temp"], df_RT["I_Ryx"])
        ax[1][0].set_ylabel("I_Rxx (A)")
        ax[1][1].set_ylabel("I_Ryx (A)")
        ax[1][0].set_xlabel("T (K)")
        ax[1][1].set_xlabel("T (K)")
    return fig
