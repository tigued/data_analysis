import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import openpyxl

# import util_tr_new as tr
import os

from scipy import interpolate  # , signal
from scipy.optimize import curve_fit
import os.path

# import sys
from pathlib import Path
from typing import List, Dict
from igor import Igor
import util_tr_new as tr
import math


def main(sample: str, dirname: str, ch_num: List[int], epilog_path="/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx", split_hall_by_theta=False, temp_convert=False):
    """_summary_

    Args:
        sample (str): _description_
        dirname (str): _description_
        ch_num (List[int]): _description_
        epilog_path (str, optional): _description_. Defaults to "/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx".
        split_hall_by_theta (bool, optional): _description_. Defaults to False.
        temp_convert (bool, optional): _description_. Defaults to False.
    """
    pwd = Path(f"/Users/uedataiga/Desktop/grad-research/data_analysis/Data/PPMS/{dirname}")
    if temp_convert:
        log_path = pwd / r"log.dat"
    else:
        log_path = None

    thickness = 0  # 単位は[m]
    temp_threshold = 2
    if os.path.exists(pwd / r"RT.dat"):
        rt = PPMS(pwd / r"RT.dat", ch_num, sample, thickness=thickness, log_path=log_path)
        rt.run_rt()
    else:
        pass
    if os.path.exists(pwd / r"Hall.dat"):
        hall = PPMS(pwd / r"Hall.dat", ch_num, sample, thickness, temp_threshold, log_path=log_path, epilog_path=epilog_path)
        hall.run_hall(split_by_theta=split_hall_by_theta)
    else:
        pass
    if os.path.exists(pwd / r"Rot.dat"):
        rot = PPMS(pwd / r"Rot.dat", ch_num, sample, thickness, temp_threshold, log_path=log_path, epilog_path=epilog_path)
        rot.run_rot()
    else:
        pass


class PPMS:
    def __init__(self, dat_path, ch_num, sample, thickness=0, temp_threshold=2, save_raw=False, log_path=None, epilog_path=None):
        self.dat_path = dat_path
        self.ch_num = ch_num
        self.sample = sample
        # self.sample_origin = "#1-1354"  # エクセルの行数を指定するのに必要
        self.thickness = thickness
        self.temp_threshold = temp_threshold
        self.log_path = log_path
        self.epilog_path = epilog_path
        self.save_raw = save_raw

        # 保存ファイルパスを決定
        self.filename, _, self.savepath = tr.make_savepath(self.dat_path, savedir=self.sample)
        self.df = pd.read_csv(self.dat_path, skiprows=range(0, 31))  # headerを飛ばしてdatをcsvとして読み込み
        # ファイル取得 dat, ch_numに対応する列を取得\
        self.time_stamp, self.B, self.temp, self.theta, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx = tr.get_arrays_from_ppms_df(self.df, self.ch_num)

        if self.log_path is not None:
            log_df = pd.read_csv(self.log_path, skiprows=range(0, 18))
            log_time_stamp, corres_temp = tr.get_arrays_from_log(log_df, ["Time Stamp (sec)", "H_Rotator (K)"])
            del log_df
            log_idx = tr.get_correspond_index(self.time_stamp, log_time_stamp)
            corres_temp = corres_temp[log_idx]

            self.temp = corres_temp
            del log_time_stamp, corres_temp
        else:
            del self.time_stamp

    def run_rot(self):  # analyze_Rot_multi_temps_ppms(self):
        print("========Rot========")

        # 各温度点にsplit
        # split_dfs = tr.split_arrays(self.temp, [self.B, self.theta, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx], self.temp_threshold, primary_array_name="temp", seconday_array_names=["B", "theta", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])

        split_dfs = tr.split_arrays(self.temp, [self.B, self.theta, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx], self.temp_threshold, primary_array_name="temp", seconday_array_names=["B", "theta", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])
        analyzed_dfs, summaries = self.analyze_Rot_multi_temps(split_dfs)
        # ファイル保存
        tr.save_dfs(analyzed_dfs, self.savepath, "temp", "Rot_", "K")  # 解析データ
        if self.save_raw:
            tr.save_dfs(split_dfs.values, self.savepath, "temp", "raw_", "K")
        summaries.to_csv(os.path.join(self.savepath, "temp_dep.csv"), index=False)  # summary

        # グラフ描画(解析データ)
        # _ = self.plot_Rot_AMR(dic=dic, dic_raw=dic_raw, savepath=os.path.join(self.savepath, "AMR.png"))
        # plt.show()

        # グラフ描画(summaryデータ)
        # savepath = os.path.join(savepath, "summary.png")
        # _ = self.plot_(df_summary, savepath=os.path.join(self.savepath, "summary.png"))
        # plt.show()

        # ログ保存
        line = "filepath: {0}\n ch_num: {1}".format(self.dat_path, self.ch_num)
        logfilepath_Rot = os.path.join(self.savepath, "logfile_Rot.txt")
        with open(logfilepath_Rot, mode="w") as f:
            f.write(line)
        return

    def run_hall(self, split_by_theta=False):  # analyze_Hall_multi_temps_ppms(self):
        if split_by_theta:
            # Rot measurementの時
            split_dfs_theta = tr.split_arrays(self.theta, [self.B, self.temp, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx], self.temp_threshold, primary_array_name="theta", seconday_array_names=["B", "temp", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])
            for theta, df in split_dfs_theta.items():
                print(f"Hall for theta = {theta} deg")
                # 各温度点にsplit
                split_dfs = tr.split_arrays(df["temp"].values, [df["B"].values, df["Rxx"].values, df["Ryx"].values, df["I_Rxx"].values, df["I_Ryx"].values], self.temp_threshold, primary_array_name="temp", seconday_array_names=["B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])

                # Hall解析
                # NOTE: summariesも出力するが，Rot measurementでは磁場の方向がc軸垂直なので意味のないデータ
                analyzed_dfs, summaries = self.analyze_Hall_multi_temps(split_dfs)

                tr.save_dfs(analyzed_dfs, self.savepath, "temp", f"Hall_{int(theta)}deg_", "K")  # 解析データ
                if self.save_raw:
                    tr.save_dfs(split_dfs.values, self.savepath, "temp", f"raw_{int(theta)}deg_", "K")  # 生データ(分割したそのまま)
                summaries.to_csv(os.path.join(self.savepath, f"temp_dep_{int(theta)}deg.csv"), index=False)  # summary

                # グラフ描画(解析データ)
                _ = self.plot_Hall_B_dep_data(analyzed_dfs, split_dfs, os.path.join(self.savepath, f"B-dep_{int(theta)}deg.png"))
                plt.show()

                # ログ保存
                line = "filepath: {0}\n ch_num: {1}".format(self.dat_path, self.ch_num)
                logfilepath_Hall = os.path.join(self.savepath, f"logfile_Hall_{int(theta)}deg.txt")
                with open(logfilepath_Hall, mode="w") as f:
                    f.write(line)
        else:
            # 各温度点にsplit
            split_dfs = tr.split_arrays(self.temp, [self.B, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx], self.temp_threshold, primary_array_name="temp", seconday_array_names=["B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"])

            # 各温度点の解析
            # dic, dic_raw, df_summary = self.analyze_Hall_multi_temps(self.temp, self.B, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx)
            # Hall解析
            analyzed_dfs, summaries = self.analyze_Hall_multi_temps(split_dfs)

            # dfs, summariesを保存
            tr.save_dfs(analyzed_dfs, self.savepath, "temp", "Hall_", "K")  # 解析データ
            if self.save_raw:
                tr.save_dfs(split_dfs.values, self.savepath, "temp", "raw_", "K")  # 生データ(分割したそのまま)
            summaries.to_csv(os.path.join(self.savepath, "temp_dep.csv"), index=False)  # summary

            # グラフ描画(解析データ)
            _ = self.plot_Hall_B_dep_data(analyzed_dfs, split_dfs, os.path.join(self.savepath, "B-dep.png"))
            plt.show()

            # グラフ描画(summaryデータ)
            _ = self.plot_Hall_summary_data(summaries, savepath=os.path.join(self.savepath, "summary.png"))
            plt.show()

            # ログ保存
            line = "filepath: {0}\n ch_num: {1}".format(self.dat_path, self.ch_num)
            logfilepath_Hall = os.path.join(self.savepath, "logfile_Hall.txt")
            with open(logfilepath_Hall, mode="w") as f:
                f.write(line)

        # df_summaryで，2Kに近い温度のcarrier_2Dとmobilityを取得，epilog_pathに保存
        # df_summary_2K = summaries[summaries["temps"] < 2.5].head(1)
        # epilogにmobility, carrier2Dを保存
        # df_epilog = pd.read_excel(self.epilog_path, sheet_name="log", header=0, index_col=None)  # こちらは保存する時エクセル全体を上書きしてしまうので非推奨
        # excel_row = int(self.sample) - int(self.sample_origin.split("-")[1]) + 1
        # cell_carrier = f"X{excel_row}"
        # cell_mobility = f"W{excel_row}"
        # print(cell_carrier, cell_mobility)
        # wb = openpyxl.load_workbook(self.epilog_path)
        # sheet = wb["log"]
        # sheet[cell_carrier] = df_summary_2K["carrier2D"].values[0]
        # sheet[cell_mobility] = df_summary_2K["mobility"].values[0]
        # print(sheet[cell_carrier], sheet[cell_mobility])
        # wb.save(self.epilog_path)
        # df_epilog[df_epilog["sample"] == self.sample]["carrier at 2K (cm^-2)"] = df_summary_2K["carrier2D"].values[0]
        # df_epilog[df_epilog["sample"] == self.sample]["mobility at 2K (cm^2V^-1s^-1)"] = df_summary_2K["mobility"].values[0]
        # df_epilog.to_excel(self.epilog_path, sheet_name="log", index=False)

        return

    def run_rt(self):  # analyze_RT_ppms(self):
        df_RT = pd.DataFrame(np.array([self.temp, self.B, self.Rxx, self.Ryx, self.I_Rxx, self.I_Ryx]).T)
        df_RT.columns = ["temp", "B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"]
        # 3Dの抵抗を表記
        if self.thickness != 0:
            df_RT["Rxx_3D"] = df_RT["Rxx"] * self.thickness * 1e2
            df_RT["Ryx_3D"] = df_RT["Ryx"] * self.thickness * 1e2
        # ファイル出力
        df_RT.to_csv(os.path.join(self.savepath, self.filename + ".csv"), index=False)
        # グラフ描画
        _ = self.plot_RT_data(df_RT, os.path.join(self.savepath, self.filename + ".png"))
        return

    def analyze_Hall_multi_temps(self, split_dfs):
        """ """
        analyzed_dfs = {}  # 各温度における解析済みデー
        # summaries = pd.DataFrame()
        for idx, (fixed_temp, df) in enumerate(split_dfs.items()):
            _ = df["temp"].values
            B = df["B"].values
            Rxx = df["Rxx"].values
            Ryx = df["Ryx"].values

            # Hall解析
            analyzed_df, summary = self.analyze_Hall(B, Rxx, Ryx, fixed_temp)
            analyzed_dfs[fixed_temp] = analyzed_df
            # summaryデータのdfを更新
            # summaries = pd.concat([summaries, summary], axis=1)

            if idx == 0:
                summaries = pd.DataFrame(columns=summary.index.tolist())

            summaries = pd.concat(
                [summaries, summary.set_axis(summaries.columns).to_frame().T],
                ignore_index=True,
                axis=0,
            )
        # # summariesを転置
        # summaries = summaries.T.reset_index(drop=True)

        # ######################
        # 昔の処理 → hoge.py
        # ######################

        # summaryデータの処理
        # すべてがNaNな行、列を削除
        summaries = summaries.dropna(how="all").dropna(how="all", axis=1)
        # RyxA_normの規格化
        if "RyxA_norm" in summaries.columns:
            summaries["RyxA_norm"] = summaries["RyxA_norm"].map(lambda x: x / summaries["RyxA_norm"][0])

        return analyzed_dfs, summaries

    def analyze_Hall(self, B_raw, Rxx_raw, Ryx_raw, fixed_temp=0):
        # 温度固定、磁場スイープ
        # 折返し有り無し判定(磁性体か非磁性か)
        is_up_down = True if B_raw[0] * B_raw[len(B_raw) - 1] > 0 else False
        if is_up_down:
            # 往復あり(磁性体)
            B_ref_u, Rxx_u, _, B_ref_d, Rxx_d, _ = tr.symmetrize(*tr.split_up_down_scans(B_raw, Rxx_raw))
            B_ref_u, _, Ryx_u, B_ref_d, _, Ryx_d = tr.symmetrize(*tr.split_up_down_scans(B_raw, Ryx_raw))
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
            # Ryx_dとRyx_uのB = 0 Tの絶対値の平均
            RyxA = (Ryx_d_int(0) - Ryx_u_int(0)) / 2
            Rxx0 = Rxx_d_int(0)  # Rxx_dのB = 0 Tの値（ダウンスイープ）
            HallAngle_d = Ryx_d / Rxx_d
            HallAngle_u = Ryx_u / Rxx_u
            HallAngle0 = RyxA / Rxx0
            # down scanのB>0だけ取り出してフィッティングする
            # subdf_d_pos = pd.DataFrame(np.array([B_raw, Ryx_raw]).T, columns=["B", "Ryx"]).query("B > 5")
            # NOTE: 上のだと生データを使っている&低磁場付近で変な挙動があると不正確になる
            # 下のように，B > 5 T以上のデータがあるときは B > 5 Tの範囲でフィッティング，それ以外は B > 0 Tの範囲でフィッティング．使用するのは反対称化したデータのダウンスキャンのみ
            if np.max(B_ref_d) < 5:
                subdf_d_pos = pd.DataFrame(np.array([B_ref_d, Ryx_d]).T, columns=["B", "Ryx"]).query("B > 0")
            else:
                print(f"use B > 5 T at {fixed_temp} K")
                subdf_d_pos = pd.DataFrame(np.array([B_ref_d, Ryx_d]).T, columns=["B", "Ryx"]).query("B > 5")
            B_ref_d_pos = subdf_d_pos["B"].values
            Ryx_d_pos = subdf_d_pos["Ryx"].values
            fit = np.polyfit(B_ref_d_pos, Ryx_d_pos, 1)  #

            carrier2D = 1e-4 / (1.602e-19 * fit[0])
            mobility = 1e4 * fit[0] / Rxx0
            if self.thickness == 0:
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
                series_summary = pd.Series(
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
                    ],
                )
            else:
                # 3D
                Rxx_u_3D = Rxx_u * self.thickness * 1e2
                Rxx_d_3D = Rxx_d * self.thickness * 1e2
                Ryx_u_3D = Ryx_u * self.thickness * 1e2
                Ryx_d_3D = Ryx_d * self.thickness * 1e2
                Gxx_d_3D = Gxx_d / (self.thickness * 1e2)
                Gxy_d_3D = Gxy_d / (self.thickness * 1e2)
                Gxx_u_3D = Gxx_u / (self.thickness * 1e2)
                Gxy_u_3D = Gxy_u / (self.thickness * 1e2)
                RyxA_3D = RyxA * self.thickness * 1e2
                Rxx0_3D = Rxx0 * self.thickness * 1e2
                carrier3D = carrier2D / (self.thickness * 1e2)
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
                series_summary = pd.Series(
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
            B_ref, Rxx, _ = self.symmetrize(B_raw, Rxx_raw)
            B_ref, _, Ryx = self.symmetrize(B_raw, Ryx_raw)
            Gxx = Rxx / (Rxx**2 + Ryx**2)
            Gxy = Ryx / (Rxx**2 + Ryx**2)
            Rxx_int = interpolate.interp1d(B_ref, Rxx)
            # Ryx_int = interpolate.interp1d(B_ref, Ryx)
            temp = np.full_like(B_ref, fixed_temp)
            Rxx0 = Rxx_int(0)
            # Ryx0 = Ryx_int(0)
            HallAngle = Ryx / Rxx
            # NOTE: 途中で量子ホール化する時はその部分を除外しないと正しい値が出ない
            # NOTE: こちらもB > 5に制限すべきか？
            try:
                fit = np.polyfit(B_ref, Ryx, 1)
            except BaseException:
                print(str(fixed_temp) + "K")
                import traceback

                traceback.print_exc()
            carrier2D = 1e-4 / (1.602e-19 * fit[0])
            mobility = 1e4 * fit[0] / Rxx0
            if self.thickness == 0:
                # 2D
                data = np.array([temp, B_ref, Rxx, Ryx, Gxx, Gxy, HallAngle, np.abs(HallAngle)])
                columns = [
                    "temp",
                    "B_ref",
                    "Rxx",
                    "Ryx",
                    "Gxx",
                    "Gxy",
                    "HallAngle",
                    "HallAngle_abs",
                ]
                series_summary = pd.Series(
                    [
                        fixed_temp,
                        carrier2D,
                        mobility,
                        np.abs(carrier2D),
                        np.abs(mobility),
                    ],
                    index=[
                        "temps",
                        "carrier2D",
                        "mobility",
                        "carrier2D_abs",
                        "mobility_abs",
                    ],
                )
            else:
                # 3D
                Rxx_3D = Rxx * self.thickness * 1e2
                Ryx_3D = Ryx * self.thickness * 1e2
                Gxx_3D = Gxx / (self.thickness * 1e2)
                Gxy_3D = Gxy / (self.thickness * 1e2)
                Rxx0_3D = Rxx0 * self.thickness * 1e2
                carrier3D = carrier2D / (self.thickness * 1e2)
                data = np.array(
                    [
                        temp,
                        B_ref,
                        Rxx,
                        Ryx,
                        Gxx,
                        Gxy,
                        HallAngle,
                        np.abs(HallAngle),
                        Rxx_3D,
                        Ryx_3D,
                        Gxx_3D,
                        Gxy_3D,
                    ]
                )
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
                series_summary = pd.Series(
                    [
                        fixed_temp,
                        carrier2D,
                        mobility,
                        np.abs(carrier2D),
                        np.abs(mobility),
                        carrier3D,
                        np.abs(carrier3D),
                    ],
                    index=[
                        "temps",
                        "carrier2D",
                        "mobility",
                        "carrier2D_abs",
                        "mobility_abs",
                        "carrier3D",
                        "carrier3D_abs",
                    ],
                )
        df = pd.DataFrame(data=data.T, columns=columns)
        return (df, series_summary)

    def analyze_Rot_multi_temps(self, split_dfs):
        """ """
        analyzed_dfs = {}  # 各温度における解析済みデー
        # summaries = pd.DataFrame()  # 各温度点の特徴量をまとめたデータ(indexは温度点)
        for idx, (fixed_temp, df) in enumerate(split_dfs.items()):
            print(f"========fixed_temp: {fixed_temp} K========")
            _ = df["temp"].values
            # B = df["B"].values
            theta = df["theta"].values
            Rxx = df["Rxx"].values
            Ryx = df["Ryx"].values

            # Rot解析
            analyzed_df, summary = self.analyze_Rot(theta, Rxx, Ryx, fixed_temp)
            analyzed_dfs[fixed_temp] = analyzed_df
            # summaryデータのdfを更新
            # summaries = pd.concat([summaries, summary], axis=1)

            # summaryデータのdfを更新
            if idx == 0:
                summaries = pd.DataFrame(columns=summary.index.tolist())
            summaries = pd.concat(
                [summaries, summary.set_axis(summaries.columns).to_frame().T],
                ignore_index=True,
                axis=0,
            )
        # # summariesを転置
        # summaries = summaries.T.reset_index(drop=True)

        # ######################
        # 昔の処理 → hoge.py
        # ######################

        # summaryデータの処理
        # すべてがNaNな行、列を削除
        summaries = summaries.dropna(how="all").dropna(how="all", axis=1)
        # RyxA_normの規格化
        if "RyxA_norm" in summaries.columns:
            summaries["RyxA_norm"] = summaries["RyxA_norm"].map(lambda x: x / summaries["RyxA_norm"][0])

        return analyzed_dfs, summaries

    def analyze_Rot(self, theta_raw, Rxx_raw, Ryx_raw, fixed_temp=0):
        # 温度固定，角度スイープ
        theta_ref_u, Rxx_u, _, theta_ref_d, Rxx_d, _ = tr.symmetrize(*tr.split_up_down_scans(theta_raw, Rxx_raw))  # 対称化
        theta_ref_u, _, Ryx_u, theta_ref_d, _, Ryx_d = tr.symmetrize(*tr.split_up_down_scans(theta_raw, Ryx_raw))  # 反対称化
        Gxx_d = Rxx_d / (Rxx_d**2 + Ryx_d**2)
        Gxy_d = Ryx_d / (Rxx_d**2 + Ryx_d**2)
        Gxx_u = Rxx_u / (Rxx_u**2 + Ryx_u**2)
        Gxy_u = Ryx_u / (Rxx_u**2 + Ryx_u**2)
        Rxx_d_int = interpolate.interp1d(theta_ref_d, Rxx_d)
        Rxx_u_int = interpolate.interp1d(theta_ref_u, Rxx_u)
        Ryx_d_int = interpolate.interp1d(theta_ref_d, Ryx_d)
        Ryx_u_int = interpolate.interp1d(theta_ref_u, Ryx_u)

        temp = np.full_like(theta_ref_u, fixed_temp)
        # Ryx_dとRyx_uのtheta = 0 Tの絶対値の平均
        # RyxA = (Ryx_d_int(0) - Ryx_u_int(0)) / 2
        # Rxx0 = Rxx_d_int(0)  # Rxx_dのtheta = 0 Tの値（ダウンスイープ）
        # Rxxのdown scanのtheta>0だけ取り出してcos^2(θ）でフィッティングする
        # subdf_d_pos = pd.DataFrame(np.array([theta_ref_d, Rxx_d]).T, columns=["theta", "Rxx"]).query("theta > 0")
        # theta_ref_d_pos = subdf_d_pos["theta"].values

        def func(x, a, b):
            return a * np.cos(np.radians(x)) ** 2 + b

        param_d, cov_d = curve_fit(func, theta_ref_d, Rxx_d)
        # プロット
        plt.plot(theta_ref_d, Rxx_d, color="black", marker="o")
        plt.plot(theta_ref_d, func(theta_ref_d, param_d[0], param_d[1]), color="blue")
        plt.xlim(0, 360)
        plt.xlabel(r"$\theta$ (deg)")
        plt.ylabel(r"$\rho_{xx}$ (Ω)")
        savepath = os.path.join(self.savepath, "2K_theta_dep.png")
        if fixed_temp < 2.5:
            plt.savefig(
                savepath,
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                pad_inches=1,
            )
        plt.show()
        plt.plot(theta_ref_d, Ryx_d, color="black", marker="o")
        plt.xlabel(r"$\theta$ (deg)")
        plt.ylabel(r"$\rho_{yx}$ (Ω)")
        plt.show()

        subdf_u_pos = pd.DataFrame(np.array([theta_ref_u, Rxx_u]).T, columns=["theta", "Rxx"]).query("theta > 0")
        theta_ref_u_pos = subdf_u_pos["theta"].values
        Rxx_u_pos = subdf_u_pos["Rxx"].values

        param_u, cov_u = curve_fit(func, theta_ref_u_pos, Rxx_u_pos)
        print(f"param: {param_u}")
        print(f"cov: {cov_u}")
        # プロット
        # plt.plot(theta_ref_u_pos, Rxx_u_pos, color="black")
        # plt.plot(theta_ref_u_pos, func(theta_ref_u_pos, param[0], param[1]), color="blue")
        # plt.show()

        # フィッティングの最大ピーク位置，最小ピーク位置を取得
        theta_max = theta_ref_d[np.argmax(func(theta_ref_d, param_d[0], param_d[1]))]
        theta_min = theta_ref_d[np.argmin(func(theta_ref_d, param_d[0], param_d[1]))]
        if abs(theta_max) > abs(theta_min):
            theta_0deg, theta_90deg = theta_max, theta_min
        else:
            theta_0deg, theta_90deg = theta_min, theta_max

        Rxx_d_fit = func(theta_ref_d, param_d[0], param_d[1])
        Rxx_u_fit = func(theta_ref_u, param_u[0], param_u[1])
        AMR_d = (Rxx_d_int(theta_ref_d) - Rxx_d_int(theta_0deg)) / Rxx_d_int(theta_0deg)
        AMR_u = (Rxx_u_int(theta_ref_d) - Rxx_u_int(theta_0deg)) / Rxx_u_int(theta_0deg)
        AMR_d_fit = (func(theta_ref_d, param_d[0], param_d[1]) - func(theta_0deg, param_d[0], param_d[1])) / func(theta_0deg, param_d[0], param_d[1])
        AMR_u_fit = (func(theta_ref_u, param_u[0], param_u[1]) - func(theta_0deg, param_u[0], param_u[1])) / func(theta_0deg, param_u[0], param_u[1])

        Rxx_0deg = Rxx_d_int(theta_0deg)
        Rxx_90deg = Rxx_d_int(theta_90deg)
        Rxx_0deg_fit = func(theta_0deg, param_d[0], param_d[1])
        Rxx_90deg_fit = func(theta_90deg, param_d[0], param_d[1])
        AMR_vertical = (Rxx_90deg - Rxx_0deg) / Rxx_0deg
        AMR_vertical_fit = (Rxx_90deg_fit - Rxx_0deg_fit) / Rxx_0deg_fit

        if self.thickness == 0:
            # 2D
            data = np.array(
                [
                    temp,
                    theta_ref_d,
                    theta_ref_u,
                    Rxx_d,
                    Rxx_d_fit,
                    Rxx_u,
                    Rxx_u_fit,
                    Ryx_d,
                    Ryx_u,
                    Gxx_d,
                    Gxx_u,
                    Gxy_d,
                    Gxy_u,
                    AMR_d,
                    AMR_u,
                    AMR_d_fit,
                    AMR_u_fit,
                ]
            )
            columns = [
                "temp",
                "theta_ref_d",
                "theta_ref_u",
                "Rxx_d",
                "Rxx_d_fit",
                "Rxx_u",
                "Rxx_u_fit",
                "Ryx_d",
                "Ryx_u",
                "Gxx_d",
                "Gxx_u",
                "Gxy_d",
                "Gxy_u",
                "AMR_d",
                "AMR_u",
                "AMR_d_fit",
                "AMR_u_fit",
            ]
            series_summary = pd.Series(
                [fixed_temp, theta_0deg, theta_90deg, Rxx_0deg, Rxx_90deg, Rxx_0deg_fit, Rxx_90deg_fit, AMR_vertical, AMR_vertical_fit],
                index=["temps", "theta_0deg", "theta_90deg", "Rxx_0deg", "Rxx_90deg", "Rxx_0deg_fit", "Rxx_90deg_fit", "AMR", "AMR_fit"],
            )
        else:
            # 3D
            Rxx_u_3D = Rxx_u * self.thickness * 1e2
            Rxx_u_3D_fit = Rxx_u_fit * self.thickness * 1e2
            Rxx_d_3D = Rxx_d * self.thickness * 1e2
            Rxx_d_3D_fit = Rxx_d_fit * self.thickness * 1e2
            Ryx_u_3D = Ryx_u * self.thickness * 1e2
            Ryx_d_3D = Ryx_d * self.thickness * 1e2
            Gxx_d_3D = Gxx_d / (self.thickness * 1e2)
            Gxy_d_3D = Gxy_d / (self.thickness * 1e2)
            Gxx_u_3D = Gxx_u / (self.thickness * 1e2)
            Gxy_u_3D = Gxy_u / (self.thickness * 1e2)
            Rxx_0deg_3D = Rxx_0deg * self.thickness * 1e2
            Rxx_90deg_3D = Rxx_90deg * self.thickness * 1e2
            Rxx_0deg_fit_3D = Rxx_0deg_fit * self.thickness * 1e2
            Rxx_90deg_fit_3D = Rxx_90deg_fit * self.thickness * 1e2
            data = np.array(
                [
                    temp,
                    theta_ref_d,
                    theta_ref_u,
                    Rxx_d,
                    Rxx_d_fit,
                    Rxx_u,
                    Rxx_u_fit,
                    Ryx_d,
                    Ryx_u,
                    Gxx_d,
                    Gxx_u,
                    Gxy_d,
                    Gxy_u,
                    AMR_d,
                    AMR_u,
                    AMR_d_fit,
                    AMR_u_fit,
                    Rxx_u_3D,
                    Rxx_u_3D_fit,
                    Rxx_d_3D,
                    Rxx_d_3D_fit,
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
                "theta_ref_d",
                "theta_ref_u",
                "Rxx_d",
                "Rxx_d_fit",
                "Rxx_u",
                "Rxx_u_fit",
                "Ryx_d",
                "Ryx_u",
                "Gxx_d",
                "Gxx_u",
                "Gxy_d",
                "Gxy_u",
                "AMR_d",
                "AMR_u",
                "AMR_d_fit",
                "AMR_u_fit",
                "Rxx_u_3D",
                "Rxx_u_3D_fit",
                "Rxx_d_3D",
                "Rxx_d_3D_fit",
                "Ryx_u_3D",
                "Ryx_d_3D",
                "Gxx_u_3D",
                "Gxx_d_3D",
                "Gxy_u_3D",
                "Gxy_d_3D",
            ]
            series_summary = pd.Series(
                [fixed_temp, theta_0deg, theta_90deg, Rxx_0deg_3D, Rxx_90deg_3D, Rxx_0deg_fit_3D, Rxx_90deg_fit_3D, AMR_vertical, AMR_vertical_fit],
                index=["temps", "theta_0deg", "theta_90deg", "Rxx_0deg_3D", "Rxx_90deg_3D", "Rxx_0deg_fit_3D", "Rxx_90deg_fit_3D", "AMR", "AMR_fit"],
            )
        df = pd.DataFrame(data=data.T, columns=columns)
        return (df, series_summary)

    def plot_Hall_summary_data(self, df_summary, savepath):
        print("==================plot summary=========================")
        # グラフ描画準備
        igor = Igor(digits=2)
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 12))
        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.3)

        # プロット:temp-dep
        if "carrier2D_abs" in df_summary.columns:
            igor.plot(
                ax=ax[0],
                xs=[df_summary["temps"].values],
                ys=[df_summary["carrier2D_abs"].values],
                xlabel=r"",
                ylabel=r"carrier 2D($\mathrm{cm^{-2}}$)",
                min_x=0,
                max_x=300,
                x_step=4,
                omit_tick="x",
            )
        elif "carrier3D_abs" in df_summary.columns:
            igor.plot(
                ax=ax[0],
                xs=[df_summary["temps"].values],
                ys=[df_summary["carrier3D_abs"].values],
                xlabel=r"",
                ylabel=r"carrier 3D($\mathrm{cm^{-3}}$)",
                min_x=0,
                max_x=300,
                x_step=4,
                omit_tick="x",
            )
        if "mobility" in df_summary.columns:
            igor.plot(
                ax=ax[1],
                xs=[df_summary["temps"].values],
                ys=[df_summary["mobility"].values],
                xlabel=r"",
                ylabel=r"mobility ($\mathrm{cm^2 V^{-1} s^{-1}}$)",
                min_x=0,
                max_x=300,
                x_step=4,
                omit_tick="x",
                savepath=savepath,
            )
        if "RyxA" in df_summary.columns:
            igor.plot(
                ax=ax[2],
                xs=[df_summary["temps"].values],
                ys=[df_summary["RyxA"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$\rho_{yx}$A (Ω)",
                min_x=0,
                max_x=300,
                x_step=4,
                savepath=savepath,
            )
        else:
            pass
        # ax[2].set_xlabel(r"$T$ (K)")
        return fig

    def plot_Hall_B_dep_data(self, analyzed_dfs: Dict[float, pd.DataFrame], split_dfs: Dict[float, pd.DataFrame], savepath: str):
        """summaryデータをプロットする"""

        def _plot(
            igor,
            ax,
            xs,
            xlim,
            ys,
            ylabel,
            yscale,
            labels=[""],
            savepath="",
            xlabel=r"",
            label_pos=(0, 0),
        ):
            if labels == [""]:
                labels = [""] * len(xs)
            if xlabel == "":
                omit_thick = "x"
            else:
                omit_thick = ""
            igor.plot(
                ax=ax,
                xs=xs,
                ys=ys,
                xlabel=xlabel,
                ylabel=ylabel,
                labels=labels,
                label_pos=label_pos,
                min_x=-xlim,
                max_x=xlim,
                scale_y=yscale,
                x_step=5,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                omit_tick=omit_thick,
                savepath=savepath,
            )
            return None

        print("==================plot B  dep=========================")
        ylabels_R = [
            r"$\rho_{xx}$ (kΩ)",
            r"$\rho_{yx}$ (kΩ)",
            r"$\rho_{xx}$ raw (kΩ)",
            r"$\rho_{yx}$ raw (kΩ)",
        ]
        y_scale_R = 1e-3
        ylabels_G = [
            r"$\sigma_{xx}$ ($e^2/h$)",
            r"$\sigma_{xy}$ ($e^2/h$)",
            r"$\sigma_{xx}$ raw ($e^2/h$)",
            r"$\sigma_{xy}$ raw ($e^2/h$)",
        ]
        y_scale_G = 6.626e-34 / (1.602e-19) ** 2  # h/e^2

        # グラフ描画準備
        igor = Igor(fontsize=30, digits=2)
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24, 20))
        fig.subplots_adjust(wspace=0.7, hspace=0.35)
        # fig.text(0.1, 0.9, sample_str, fontsize=25)

        temps, xs, Rxx, Ryx, Gxx, Gxy, Ryx_d_sq, BoverRyx = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for temp, df in analyzed_dfs.items():
            temps.append(temp)
            # プロット:解析データ
            if "B_ref_d" in df.columns:
                # AHE
                xs.append(pd.concat([df["B_ref_d"], df["B_ref_u"]]).values)
                Rxx.append(pd.concat([df["Rxx_d"], df["Rxx_u"]]).values)
                Ryx.append(pd.concat([df["Ryx_d"], df["Ryx_u"]]).values)
                Gxx.append(pd.concat([df["Gxx_d"], df["Gxx_u"]]).values)
                Gxy.append(pd.concat([df["Gxy_d"], df["Gxy_u"]]).values)
                Ryx_d_sq.append(df["Ryx_d_sq"].values)
                BoverRyx.append(df["B_over_Ryx"].values)
            else:
                # OHE (Ordinary Hall effect ?)
                xs.append(df["B_ref"].values)
                Rxx.append(df["Rxx"].values)
                Ryx.append(df["Ryx"].values)
                Gxx.append(df["Gxx"].values)
                Gxy.append(df["Gxy"].values)
                Ryx_d_sq.append([])
                BoverRyx.append([])

        # 2Kのデータをプロット
        idx_2K = np.argmin(np.abs(np.array(temps).astype(float) - 2))
        x_2K = np.array(xs[idx_2K])
        _plot(
            igor,
            ax[0][0],
            [x_2K.copy()],
            10,
            [Rxx[idx_2K].copy()],
            ylabels_R[0],
            y_scale_R,
            # savepath=savepath,
        )
        _plot(
            igor,
            ax[1][0],
            [x_2K.copy()],
            10,
            [Ryx[idx_2K].copy()],
            ylabels_R[1],
            y_scale_R,
        )
        _plot(
            igor,
            ax[2][0],
            [x_2K.copy()],
            10,
            [Gxx[idx_2K].copy()],
            ylabels_G[0],
            y_scale_G,
        )
        _plot(
            igor,
            ax[3][0],
            [x_2K.copy()],
            10,
            [Gxy[idx_2K].copy()],
            ylabels_G[1],
            y_scale_G,
            xlabel=r"$B$ (T)",
        )

        # 全温度のデータをプロット
        _plot(igor, ax[0][1], xs, 1, Rxx, ylabels_R[0], y_scale_R)
        _plot(igor, ax[1][1], xs, 1, Ryx, ylabels_R[1], y_scale_R)
        _plot(igor, ax[2][1], xs, 1, Gxx, ylabels_G[0], y_scale_G)
        _plot(igor, ax[3][1], xs, 1, Gxy, ylabels_G[1], y_scale_G, xlabel=r"$B$ (T)")

        xs_raw, Rxx_raw, Ryx_raw, Gxx_raw, Gxy_raw = [], [], [], [], []
        Ixx, Iyx = [], []
        temps_str = []
        for temp, df in split_dfs.items():
            if temps_str == []:
                temps_str.append(r"$T$ = " + str(temp) + " K")
            else:
                temps_str.append(str(temp) + " K")
            # プロット:raw
            xs_raw.append(df["B"])
            Rxx_raw.append(df["Rxx"])
            Ryx_raw.append(df["Ryx"])
            Gxx_raw.append(df["Rxx"] / (df["Rxx"] ** 2 + df["Ryx"] ** 2))
            Gxy_raw.append(df["Ryx"] / (df["Rxx"] ** 2 + df["Ryx"] ** 2))
            # ys_xx_raw.append(df["Gxx_raw"])
            # ys_yx_raw.append(df["Gxy_raw"])
            Ixx.append(df["I_Rxx"])
            Iyx.append(df["I_Ryx"])
        _plot(igor, ax[0][2], xs_raw, 1, Rxx_raw, ylabels_R[2], y_scale_R)
        _plot(igor, ax[1][2], xs_raw, 1, Ryx_raw, ylabels_R[3], y_scale_R)
        _plot(igor, ax[2][2], xs_raw, 1, Gxx_raw, ylabels_G[2], y_scale_G)
        _plot(
            igor,
            ax[3][2],
            xs_raw,
            1,
            Gxy_raw,
            ylabels_G[3],
            y_scale_G,
            xlabel=r"$B$ (T)",
        )
        _plot(
            igor,
            ax[0][3],
            xs_raw,
            1,
            Ixx,
            "I_Rxx (μA)",
            1e6,
            labels=temps_str,
            label_pos=(1, 1),
        )
        _plot(igor, ax[1][3], xs_raw, 1, Iyx, "I_Ryx (μA)", 1e6, xlabel=r"$B$ (T)")
        # Gyx vs Gxx
        igor.plot(
            ax=ax[2][3],
            xs=Gxy,
            ys=Gxx,
            xlabel=r"$\sigma_{xy}$ ($e^2/h$)",
            ylabel=r"$\sigma_{xx}$ ($e^2/h$)",
            labels=[""] * len(Gxy),
            scale_x=y_scale_G,
            scale_y=y_scale_G,
            x_step=3,
            y_step=3,
        )
        # Arrott plot
        igor.plot(
            ax=ax[3][3],
            xs=BoverRyx,
            ys=Ryx_d_sq,
            xlabel=r"$B / \rho_{yx}$ (a.u.)",
            ylabel=r"$\rho_{yx}^2$ (a.u.)",
            labels=[""] * len(BoverRyx),
            x_step=3,
            y_step=3,
            savepath=savepath,
        )

        return fig

    def plot_RT_data(self, df_RT, savepath=""):
        igor = Igor(digits=2)
        print("==================plot RT=========================")
        if "Rxx_3D" in df_RT.columns:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.2)
            igor.plot(
                ax=ax[0][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Rxx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{xx}$2D (kΩ)",
                min_x=0,
                max_x=300,
                min_y=0,
                max_y=None if df_RT["Rxx"].max() < 0 else 0,
                scale_y=1e-3,
                x_step=4,
            )
            igor.plot(
                ax=ax[0][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Ryx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{yx}$2D (kΩ)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Ryx"].max() < 0 else 0,
                scale_y=1e-3,
                x_step=4,
            )
            igor.plot(
                ax=ax[1][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Rxx_3D"].values],
                xlabel=r"",
                ylabel=r"$\rho_{xx}$3D (Ωcm)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Rxx_3D"].max() < 0 else 0,
                x_step=4,
            )
            igor.plot(
                ax=ax[1][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Ryx_3D"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$\rho_{yx}$3D (Ωcm)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Ryx_3D"].max() < 0 else 0,
                x_step=4,
            )
            igor.plot(
                ax=ax[2][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Rxx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{xx}$ (μA)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["I_Rxx"].max() < 0 else 0,
                scale_y=1e6,
                x_step=4,
            )
            igor.plot(
                ax=ax[2][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Ryx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{yx}$ (μA)",
                min_x=0,
                max_x=300,
                scale_y=1e6,
                x_step=4,
                savepath=savepath,
            )
        else:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            plt.subplots_adjust(wspace=0.5, hspace=0.2)
            igor.plot(
                ax=ax[0][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Rxx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{xx}$2D (kΩ)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Rxx"].max() < 0 else 0,
                scale_y=1e-3,
                x_step=4,
            )
            igor.plot(
                ax=ax[0][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Ryx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{yx}$2D (kΩ)",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Ryx"].max() < 0 else 0,
                scale_y=1e-3,
                x_step=4,
            )
            igor.plot(
                ax=ax[1][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Rxx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{xx}$ (μA)",
                min_x=0,
                max_x=300,
                scale_y=1e6,
                x_step=4,
            )
            igor.plot(
                ax=ax[1][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Ryx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{yx}$ (μA)",
                min_x=0,
                max_x=300,
                scale_y=1e6,
                x_step=4,
                savepath=savepath,
            )

        return fig

    def plot_Rot(self):
        """各角度のB依存性，2Kでのtheta依存性，AMR ratioの取得"""
        pass
