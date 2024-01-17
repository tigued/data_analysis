import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import util_tr as tr
import os.path


def make_savepath(filepath, savedirname=""):
    # 保存ファイルパスを決定
    filedirname = os.path.dirname(filepath)
    filebasename = os.path.basename(filepath)
    filename, fileext = os.path.splitext(filebasename)
    if savedirname == "":
        savepath = filedirname
    else:
        savepath = os.path.join(filedirname, savedirname)
        if os.path.isdir(savepath) == False:
            print("makedirs : " + savepath)
            os.makedirs(savepath)
    return filename, fileext, savepath


def get_arrays_from_ppms_df(df, ch_num):
    # 各々に対応するdat, ch_numに対応する列を取得
    B = df["Magnetic Field (Oe)"].values * 1e-4
    temp = df["Temperature (K)"].values
    theta = df["Sample Position (deg)"].values
    Rxx = df["Bridge " + str(ch_num[0]) + " Resistivity (Ohm-cm)"].values
    Ryx = df["Bridge " + str(ch_num[1]) + " Resistivity (Ohm-cm)"].values
    I_Rxx = df["Bridge " + str(ch_num[0]) + " Excitation (uA)"].values * 1e-6
    I_Ryx = df["Bridge " + str(ch_num[1]) + " Excitation (uA)"].values * 1e-6
    return B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx


def analyze_Hall_multi_temps_ppms(filepath, ch_num, sample_str, thickness = 0, temp_threshold = 0.1, save_raw = False):
    # 保存ファイルパスを決定
    filename, fileext, savepath = make_savepath(filepath, savedirname=sample_str)
    # df取得
    df = pd.read_csv(filepath, skiprows=range(0, 31))
    # ファイル取得 dat, ch_numに対応する列を取得
    B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx = get_arrays_from_ppms_df(df, ch_num)
    # 各温度点にsplit
    split_dfs = tr.split_arrays(temp, [B, Rxx, Ryx, I_Rxx, I_Ryx], temp_threshold, "temp", ["B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])
    # Hall解析
    analyzed_dfs, summaries = tr.do_Hall_analysis_for_dfs(split_dfs, thickness)
    # dfs, summariesを保存
    tr.save_dfs(analyzed_dfs, savepath, "temp", "Hall_", "K") #解析データ
    if save_raw:
        tr.save_dfs(split_dfs, savepath, "temp", "raw_", "K") #生データ(分割したそのまま)
    summaries.to_csv(os.path.join(savepath, "temp_dep.csv"), index=False) #summary
    # グラフ描画(解析データ)
    fig = tr.plot_B_dep_data(analyzed_dfs, split_dfs, sample_str)
    # グラフ保存(解析データ)
    fig.savefig(os.path.join(savepath, "B-dep.png"))
    # グラフ描画(summaryデータ)
    fig = tr.plot_summary_data(summaries, sample_str)
    # グラフ保存(summaryデータ)
    fig.savefig(os.path.join(savepath, "summary.png"))
    # ログ保存
    line = "filepath: {0}\nthickness: {1}[m]\nch_num: {2}".format(filepath, thickness, ch_num)
    logfilepath_Hall = os.path.join(savepath, "logfile_Hall.txt")
    with open(logfilepath_Hall, mode="w") as f:
        f.write(line)
    return


def analyze_RT_ppms(filepath, ch_num, sample_str, thickness=0):
    # 保存ファイルパスを決定
    filename, fileext, savepath = make_savepath(filepath, savedirname=sample_str)
    # df取得
    df = pd.read_csv(filepath, skiprows=range(0, 31))
    # ファイル取得 dat, ch_numに対応する列を取得
    B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx = get_arrays_from_ppms_df(df, ch_num)
    df_RT = pd.DataFrame(np.array([temp, B, Rxx, Ryx, I_Rxx, I_Ryx]).T)
    df_RT.columns = ["temp", "B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"]
    # 3Dの抵抗を表記
    if thickness != 0:
        df_RT["Rxx_3D"] = df_RT["Rxx"] * thickness * 1e2
        df_RT["Ryx_3D"] = df_RT["Ryx"] * thickness * 1e2
    # ファイル出力
    df_RT.to_csv(os.path.join(savepath, filename + ".csv"), index=False)
    # グラフ描画
    fig = tr.plot_RT_data(df_RT, sample_str)
    # グラフ保存
    fig.savefig(os.path.join(savepath, filename + ".png"))
    return
