import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

# import util_tr_new as tr
import os

from scipy import interpolate  # , signal
import os.path

# import sys
from pathlib import Path
from typing import List
from igor import Igor


def main(
    sample: str,
    dirname: str,
    ch_num: List[int],
    epilog_path="/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx",
):
    pwd = Path(f"/Users/uedataiga/Desktop/grad-research/data_analysis/Data/PPMS/{dirname}")
    # filepath_Hall = pwd / r"Hall.dat"
    filepath_Rot = pwd / r"Rot.dat"
    filepath_RT = pwd / r"RT.dat"
    thickness = 0  # 単位は[m]
    temp_threshold = 0.1

    # 実行
    rt = PPMS_RT(filepath_RT, ch_num, sample, thickness)
    rt.analyze_RT_ppms()
    # hall = PPMS_Hall(filepath_Hall, ch_num, sample, thickness, temp_threshold, epilog_path)
    # hall.analyze_Hall_multi_temps_ppms()
    rot = PPMS_Rot(filepath_Rot, ch_num, sample, thickness, temp_threshold, epilog_path)
    rot.analyze_Rot_multi_temps_ppms()


class PPMS_Rot:
    def __init__(self, filepath, ch_num, sample, thickness, temp_threshold, epilog_path):
        self.dat_path = filepath
        self.ch_num = ch_num
        self.sample = sample
        # self.sample_origin = "#1-1354"  # エクセルの行数を指定するのに必要
        self.thickness = thickness
        self.temp_threshold = temp_threshold
        self.epilog_path = epilog_path

    def run(self):  # analyze_Rot_multi_temps_ppms(self):
        # 保存ファイルパスを決定
        _, _, self.savepath = make_savepath(self.dat_path, savedir=self.sample)
        print("savepath", self.savepath)
        df = pd.read_csv(self.dat_path, skiprows=range(0, 31))  # headerを飛ばしてdatをcsvとして読み込み
        # ファイル取得 dat, ch_numに対応する列を取得\
        B, temp, sample_position, Rxx, Ryx, I_Rxx, I_Ryx = get_arrays_from_ppms_df(df, self.ch_num)
        # 解析・グラフ作成
        self.analyze_and_plot_Rot_data(temp, B, sample_position, Rxx, Ryx, I_Rxx, I_Ryx)
        # ログ保存
        line = "filepath: {0}\n ch_num: {1}".format(self.filepath, self.ch_num)
        logfilepath_Hall = os.path.join(self.savepath, "logfile_Hall.txt")
        with open(logfilepath_Hall, mode="w") as f:
            f.write(line)
        return

    def analyze_and_plot_Rot_data(self, temp, B, sample_position, Rxx, Ryx, I_Rxx=None, I_Ryx=None):
        # 各温度点の解析
        dic, dic_raw, df_summary = self.analyze_Rot_multi_temps(temp, B, sample_position, Rxx, Ryx, I_Rxx, I_Ryx)
        # ファイル保存
        self.save_Rot_multi_temps(dic, dic_raw, df_summary)

        # グラフ描画(解析データ)
        _ = self.plot_B_dep_data(dic=dic, dic_raw=dic_raw, savepath=os.path.join(self.savepath, "B-dep.png"))
        plt.show()

        # # グラフ保存(解析データ)
        # fig.savefig(os.path.join(savepath, "B-dep.png"), transparent=True)

        # グラフ描画(summaryデータ)
        # savepath = os.path.join(savepath, "summary.png")
        _ = self.plot_summary_data(df_summary, savepath=os.path.join(self.savepath, "summary.png"))
        plt.show()
        # グラフ保存(summaryデータ)
        # fig.savefig(os.path.join(savepath, "summary.png"), transparent=True)
        return None

    def analyze_Rot_multi_temps(self, temp, B, Rxx, Ryx, I_Rxx=None, I_Ryx=None):
        """_summary_

        Args:
            temp (_type_): _description_
            B (_type_): _description_
            Rxx (_type_): _description_
            Ryx (_type_): _description_
            I_Rxx (_type_, optional): _description_. Defaults to None.
            I_Ryx (_type_, optional): _description_. Defaults to None.
            thickness (int, optional): _description_. Defaults to 0.
            temp_threshold (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        if I_Rxx is None or I_Ryx is None:
            df = pd.DataFrame(np.array([temp, B, Rxx, Ryx]).T, columns=["temp", "B", "Rxx", "Ryx"])
            df_summary = pd.DataFrame(
                columns=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                ]
            )  # 各温度点の特徴量をまとめたデータ(indexは温度点)
        else:
            exists_I = True
            df = pd.DataFrame(
                np.array([temp, B, Rxx, Ryx, I_Rxx, I_Ryx]).T,
                columns=["temp", "B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"],
            )
            df_summary = pd.DataFrame(
                columns=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                    "I_Rxx",
                    "I_Ryx",
                ]
            )  # 各温度点の特徴量をまとめたデータ(indexは温度点)

        # 各温度ごとのデータをまとめるための準備(df_summary, dic, dic_raw)
        # df_summary = pd.DataFrame()  # 各温度点の特徴量をまとめたデータ(indexは温度点)
        dic = {}  # 各温度における解析済みデータ
        dic_raw = {}  # 各温度点における生データ

        # 各温度点で解析
        # 温度のdiffを取る
        df["temp_diff"] = df["temp"].diff()
        # diffがしきい値を超えたところでgroupbyする。
        # (df["temp_diff"] > temp_threshold)という列に対してcumsum(要素の足し合わせ)を実行
        for _, df_raw in df.groupby((df["temp_diff"] > self.temp_threshold).cumsum()):
            # 平均温度を計算
            fixed_temp = round(df_raw["temp"].mean(), 1)
            print(f"-------analyze_Rot_multi_temps: {fixed_temp}------")
            # 各温度で生データ取得
            # temp_raw = df_raw["temp"].values
            B_raw = df_raw["B"].values
            Rxx_raw = df_raw["Rxx"].values
            Ryx_raw = df_raw["Ryx"].values
            df_raw = df_raw.drop("temp_diff", axis=1)
            # Hall解析
            df, series_summary = self.analyze_Hall(B_raw, Rxx_raw, Ryx_raw, fixed_temp)
            # 温度点の解析結果をdicに追加
            dic[str(fixed_temp)] = df
            dic_raw[str(fixed_temp)] = df_raw
            if exists_I is True:
                # summaryにExcitation current追加
                series_summary["I_Rxx"] = df_raw["I_Rxx"].mean()
                series_summary["I_Ryx"] = df_raw["I_Ryx"].mean()
                # df_rawの列名変更
                df_raw.columns = [
                    "temp_raw",
                    "B_raw",
                    "Rxx_raw",
                    "Ryx_raw",
                    "I_Rxx",
                    "I_Ryx",
                ]
            else:
                # df_rawの列名変更
                df_raw.columns = ["temp_raw", "B_raw", "Rxx_raw", "Ryx_raw"]
            # summaryデータのdfを更新
            # df_summary = df_summary.append(series_summary, ignore_index=True)
            df_summary = pd.concat(
                [df_summary, series_summary.set_axis(df_summary.columns).to_frame().T],
                ignore_index=True,
                axis=0,
            )
        # summaryデータの処理
        # すべてがNaNな行、列を削除
        df_summary = df_summary.dropna(how="all").dropna(how="all", axis=1)
        # RyxA_normの規格化
        if "RyxA_norm" in df_summary.columns:
            df_summary["RyxA_norm"] = df_summary["RyxA_norm"].map(lambda x: x / df_summary["RyxA_norm"][0])
        return dic, dic_raw, df_summary

    def save_Rot_multi_temps(self, dic, dic_raw, df_summary):
        # 対称化データ(Hall)
        if dic is not None:
            for temp, df in dic.items():
                df.to_csv(
                    os.path.join(self.savepath, "Rot_" + str(temp) + "K.csv"),
                    index=False,
                )
        # 対称化していない生データ(raw)
        if dic_raw is not None:
            for temp, df_raw in dic_raw.items():
                df_raw.to_csv(
                    os.path.join(self.savepath, "raw_" + str(temp) + "K.csv"),
                    index=False,
                )
        # summaryデータ:temp_dep
        if df_summary is not None:
            df_summary.to_csv(os.path.join(self.savepath, "temp_dep.csv"), index=False)
        return

    def split_up_down(self, x_raw, y_raw):
        """
        データをup sweepとdown sweepに分ける
        x_raw: x_raw of sweep
        y_raw: y_raw of sweep
        """
        size = len(x_raw)
        is_sweep_up_down = False if x_raw[0] > x_raw[size // 4] else True
        # print("up to down" if(is_sweep_up_down) else "down to up")
        if is_sweep_up_down:
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

    def common_part(self, x, y):
        # 1D-arrayXとYの範囲の共通部分に属する要素のみを残す
        # 共通部分の最大値＝それぞれのarrayの最大値の中で最小のもの
        max_value = np.min([np.max(x), np.max(y)])
        min_value = np.max([np.min(x), np.min(y)])
        concat_array = np.concatenate([x, y])
        common_array = sorted(set([x for x in concat_array if x <= max_value and x >= min_value]))
        return common_array

    def symmetrize(self, x_raw_u, y_raw_u, x_raw_d=None, y_raw_d=None):
        """
        データを折り返し対称にする
        x_raw_u: x_raw of up sweep
        y_raw_u: y_raw of up sweep
        x_raw_d: x_raw of down sweep
        y_raw_d: y_raw of down sweep

        """
        if x_raw_d is None and y_raw_d is None:  # もしdown sweepがないなら
            y_int = interpolate.interp1d(x_raw_u, y_raw_u)
            x_ref = np.array([x for x in x_raw_u if x <= np.max(x_raw_u * -1) and x >= np.min(x_raw_u * -1)])
            y_sym = (y_int(x_ref) + y_int(-1 * x_ref)) / 2
            y_asym = (y_int(x_ref) - y_int(-1 * x_ref)) / 2
            return (x_ref, y_sym, y_asym)
        else:  # もしdown sweepがあるなら
            y_int_u = interpolate.interp1d(x_raw_u, y_raw_u)
            y_int_d = interpolate.interp1d(x_raw_d, y_raw_d)
            x_ref_u = self.common_part(x_raw_d, x_raw_u)
            x_ref_d = x_ref_u[::-1]
            y_sym_d = (y_int_d(x_ref_d) + y_int_u(x_ref_u)) / 2
            y_asym_d = (y_int_d(x_ref_d) - y_int_u(x_ref_u)) / 2
            # y_sym_u = y_sym_d[::-1]
            y_sym_u = y_sym_d
            # y_asym_u = y_asym_d[::-1]
            y_asym_u = y_asym_d * -1
            return (x_ref_u, y_sym_u, y_asym_u, x_ref_d, y_sym_d, y_asym_d)

    def plot_summary_data(self, df_summary, savepath=""):
        print("==================plot summary=========================")
        # グラフ描画準備
        igor = Igor(digits=2)
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 12))
        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.3)
        # fig.text(0.1, 0.9, sample_str)

        # プロット:temp-dep
        # ax[0].plot(
        #     df_summary["temps"], df_summary["carrier2D_abs"], marker="o", color="red"
        # )
        # ax[0].set_ylabel("carrier 2D(cm-2)")
        if "carrier2D_abs" in df_summary.columns:
            igor.plot(
                ax=ax[0],
                xs=[df_summary["temps"].values],
                ys=[df_summary["carrier2D_abs"].values],
                xlabel=r"",
                ylabel=r"carrier 2D($\mathrm{cm^{-2}}$)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick="x",
                title="",
                suffix="",
                savepath="",
            )

    def plot_B_dep_data(self, dic, dic_raw, savepath=""):
        """_summary_

        Args:
            dic (_type_): _description_
            dic_raw (_type_): _description_
            sample_str (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """

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
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=yscale,
                x_step=5,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick=omit_thick,
                title="",
                suffix="",
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
        for temp, df in dic.items():
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
        # 2Kに最も近い
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
            savepath=savepath,
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
        for temp, df in dic_raw.items():
            if temps_str == []:
                temps_str.append(r"$T$ = " + str(temp) + " K")
            else:
                temps_str.append(str(temp) + " K")
            # プロット:raw
            xs_raw.append(df["B_raw"])
            Rxx_raw.append(df["Rxx_raw"])
            Ryx_raw.append(df["Ryx_raw"])
            Gxx_raw.append(df["Rxx_raw"] / (df["Rxx_raw"] ** 2 + df["Ryx_raw"] ** 2))
            Gxy_raw.append(df["Ryx_raw"] / (df["Rxx_raw"] ** 2 + df["Ryx_raw"] ** 2))
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
            sub_x_step=2,
            y_step=3,
            sub_y_step=2,
            savepath="",
        )
        # Arrott plot
        igor.plot(
            ax=ax[3][3],
            xs=BoverRyx,
            ys=Ryx_d_sq,
            xlabel=r"$B / \rho_{yx}$ (a.u.)",
            ylabel=r"$\rho_{yx}^2$ (a.u.)",
            labels=[""] * len(BoverRyx),
            scale_x=1,
            scale_y=1,
            x_step=3,
            sub_x_step=2,
            y_step=3,
            sub_y_step=2,
            savepath=savepath,
        )

        return fig


class PPMS_Hall:
    def __init__(self, filepath, ch_num, sample, thickness, temp_threshold, epilog_path):
        self.filepath = filepath
        self.ch_num = ch_num
        self.sample = sample
        self.sample_origin = "#1-1354"  # エクセルの行数を指定するのに必要
        self.thickness = thickness
        self.temp_threshold = temp_threshold
        self.epilog_path = epilog_path

    def run(self):  # analyze_Hall_multi_temps_ppms(self):
        """_summary_

        Args:
            filepath (_type_):
            ch_num (_type_): _description_
            sample_str (_type_): _description_
            thickness (int, optional): _description_. Defaults to 0.
            temp_threshold (float, optional): _description_. Defaults to 0.1.
        """
        # 保存ファイルパスを決定
        _, _, self.savepath = make_savepath(self.filepath, savedir=self.sample)
        print("savepath", self.savepath)
        df = pd.read_csv(self.filepath, skiprows=range(0, 31))
        # ファイル取得 dat, ch_numに対応する列を取得\
        B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx = get_arrays_from_ppms_df(df, self.ch_num)
        # 解析・グラフ作成
        self.analyze_and_plot_Hall_data(temp, B, Rxx, Ryx, I_Rxx, I_Ryx)
        # ログ保存
        line = "filepath: {0}\n ch_num: {1}".format(self.filepath, self.ch_num)
        logfilepath_Hall = os.path.join(self.savepath, "logfile_Hall.txt")
        with open(logfilepath_Hall, mode="w") as f:
            f.write(line)
        return

    def analyze_Hall(self, B_raw, Rxx_raw, Ryx_raw, fixed_temp=0):
        # 温度固定、磁場スイープ
        # 折返し有り無し判定(磁性体か非磁性か)
        is_up_down = True if B_raw[0] * B_raw[len(B_raw) - 1] > 0 else False
        if is_up_down:
            # 往復あり(磁性体)
            B_ref_u, Rxx_u, _, B_ref_d, Rxx_d, _ = self.symmetrize(*self.split_up_down(B_raw, Rxx_raw))
            B_ref_u, _, Ryx_u, B_ref_d, _, Ryx_d = self.symmetrize(*self.split_up_down(B_raw, Ryx_raw))
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
            subdf_d_pos = pd.DataFrame(np.array([B_raw, Ryx_raw]).T, columns=["B", "Ryx"]).query("B > 0")
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

    def analyze_Hall_multi_temps(self, temp, B, Rxx, Ryx, I_Rxx=None, I_Ryx=None):
        """_summary_

        Args:
            temp (_type_): _description_
            B (_type_): _description_
            Rxx (_type_): _description_
            Ryx (_type_): _description_
            I_Rxx (_type_, optional): _description_. Defaults to None.
            I_Ryx (_type_, optional): _description_. Defaults to None.
            thickness (int, optional): _descrziption_. Defaults to 0.
            temp_threshold (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        if I_Rxx is None or I_Ryx is None:
            df = pd.DataFrame(np.array([temp, B, Rxx, Ryx]).T, columns=["temp", "B", "Rxx", "Ryx"])
            df_summary = pd.DataFrame(
                columns=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                ]
            )  # 各温度点の特徴量をまとめたデータ(indexは温度点)
        else:
            exists_I = True
            df = pd.DataFrame(
                np.array([temp, B, Rxx, Ryx, I_Rxx, I_Ryx]).T,
                columns=["temp", "B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"],
            )
            df_summary = pd.DataFrame(
                columns=[
                    "temps",
                    "Rxx0T",
                    "RyxA",
                    "RyxA_norm",
                    "carrier2D",
                    "mobility",
                    "carrier2D_abs",
                    "mobility_abs",
                    "HallAngle0",
                    "I_Rxx",
                    "I_Ryx",
                ]
            )  # 各温度点の特徴量をまとめたデータ(indexは温度点)

        # 各温度ごとのデータをまとめるための準備(df_summary, dic, dic_raw)
        # df_summary = pd.DataFrame()  # 各温度点の特徴量をまとめたデータ(indexは温度点)
        dic = {}  # 各温度における解析済みデータ
        dic_raw = {}  # 各温度点における生データ

        # 各温度点で解析
        # 温度のdiffを取る
        df["temp_diff"] = df["temp"].diff()
        # diffがしきい値を超えたところでgroupbyする。
        # (df["temp_diff"] > temp_threshold)という列に対してcumsum(要素の足し合わせ)を実行
        for _, df_raw in df.groupby((df["temp_diff"] > self.temp_threshold).cumsum()):
            # 平均温度を計算
            fixed_temp = round(df_raw["temp"].mean(), 1)
            print(f"-------analyze_Hall_multi_temps: {fixed_temp}------")
            # 各温度で生データ取得
            # temp_raw = df_raw["temp"].values
            B_raw = df_raw["B"].values
            Rxx_raw = df_raw["Rxx"].values
            Ryx_raw = df_raw["Ryx"].values
            df_raw = df_raw.drop("temp_diff", axis=1)
            # Hall解析
            analyzed_df, series_summary = self.analyze_Hall(B_raw, Rxx_raw, Ryx_raw, fixed_temp)
            # 温度点の解析結果をdicに追加
            dic[str(fixed_temp)] = analyzed_df
            dic_raw[str(fixed_temp)] = df_raw
            if exists_I is True:
                # summaryにExcitation current追加
                series_summary["I_Rxx"] = df_raw["I_Rxx"].mean()
                series_summary["I_Ryx"] = df_raw["I_Ryx"].mean()
                # df_rawの列名変更
                df_raw.columns = [
                    "temp_raw",
                    "B_raw",
                    "Rxx_raw",
                    "Ryx_raw",
                    "I_Rxx",
                    "I_Ryx",
                ]
            else:
                # df_rawの列名変更
                df_raw.columns = ["temp_raw", "B_raw", "Rxx_raw", "Ryx_raw"]
            # summaryデータのdfを更新
            # df_summary = df_summary.append(series_summary, ignore_index=True)
            df_summary = pd.concat(
                [df_summary, series_summary.set_axis(df_summary.columns).to_frame().T],
                ignore_index=True,
                axis=0,
            )
        # summaryデータの処理
        # すべてがNaNな行、列を削除
        df_summary = df_summary.dropna(how="all").dropna(how="all", axis=1)
        # RyxA_normの規格化
        if "RyxA_norm" in df_summary.columns:
            df_summary["RyxA_norm"] = df_summary["RyxA_norm"].map(lambda x: x / df_summary["RyxA_norm"][0])
        return dic, dic_raw, df_summary

    def save_Hall_multi_temps(self, dic, dic_raw, df_summary):
        # 対称化データ(Hall)
        if dic is not None:
            for temp, df in dic.items():
                df.to_csv(
                    os.path.join(self.savepath, "Hall_" + str(temp) + "K.csv"),
                    index=False,
                )
        # 対称化していない生データ(raw)
        if dic_raw is not None:
            for temp, df_raw in dic_raw.items():
                df_raw.to_csv(
                    os.path.join(self.savepath, "raw_" + str(temp) + "K.csv"),
                    index=False,
                )
        # summaryデータ:temp_dep
        if df_summary is not None:
            df_summary.to_csv(os.path.join(self.savepath, "temp_dep.csv"), index=False)
        return

    def analyze_and_plot_Hall_data(
        self,
        temp,
        B,
        Rxx,
        Ryx,
        I_Rxx=None,
        I_Ryx=None,
    ):
        # 各温度点の解析
        dic, dic_raw, df_summary = self.analyze_Hall_multi_temps(temp, B, Rxx, Ryx, I_Rxx, I_Ryx)
        # df_summaryで，2Kに近い温度のcarrier_2Dとmobilityを取得，epilog_pathに保存
        df_summary_2K = df_summary[df_summary["temps"] < 2.5].head(1)
        # df_epilog = pd.read_excel(self.epilog_path, sheet_name="log", header=0, index_col=None)  # こちらは保存する時エクセル全体を上書きしてしまうので非推奨
        excel_row = int(self.sample) - int(self.sample_origin.split("-")[1]) + 1
        cell_carrier = f"X{excel_row}"
        cell_mobility = f"W{excel_row}"
        print(cell_carrier, cell_mobility)
        wb = openpyxl.load_workbook(self.epilog_path)
        sheet = wb["log"]
        sheet[cell_carrier] = df_summary_2K["carrier2D"].values[0]
        sheet[cell_mobility] = df_summary_2K["mobility"].values[0]
        print(sheet[cell_carrier], sheet[cell_mobility])
        wb.save(self.epilog_path)

        # print(df_epilog[df_epilog["sample"] == self.sample])
        # df_epilog[df_epilog["sample"] == self.sample]["carrier at 2K (cm^-2)"] = df_summary_2K["carrier2D"].values[0]
        # df_epilog[df_epilog["sample"] == self.sample]["mobility at 2K (cm^2V^-1s^-1)"] = df_summary_2K["mobility"].values[0]
        # print(df_epilog[df_epilog["sample"] == self.sample])
        # print(f"carreir2D: {df_summary_2K['carrier2D'].values[0]}, mobility: {df_summary_2K['mobility'].values[0]}")
        # df_epilog.to_excel(self.epilog_path, sheet_name="log", index=False)
        # ファイル保存
        self.save_Hall_multi_temps(dic, dic_raw, df_summary)

        # グラフ描画(解析データ)
        # savepath =
        _ = self.plot_B_dep_data(dic=dic, dic_raw=dic_raw, savepath=os.path.join(self.savepath, "B-dep.png"))
        plt.show()

        # # グラフ保存(解析データ)
        # fig.savefig(os.path.join(savepath, "B-dep.png"), transparent=True)

        # グラフ描画(summaryデータ)
        # savepath = os.path.join(savepath, "summary.png")
        _ = self.plot_summary_data(df_summary, savepath=os.path.join(self.savepath, "summary.png"))
        plt.show()
        # グラフ保存(summaryデータ)
        # fig.savefig(os.path.join(savepath, "summary.png"), transparent=True)
        return None

    # ###################### plot ###########################
    def plot_B_dep_data(self, dic, dic_raw, savepath=""):
        """_summary_

        Args:
            dic (_type_): _description_
            dic_raw (_type_): _description_
            sample_str (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """

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
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=yscale,
                x_step=5,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick=omit_thick,
                title="",
                suffix="",
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
        for temp, df in dic.items():
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
        # 2Kに最も近い
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
            savepath=savepath,
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
        for temp, df in dic_raw.items():
            if temps_str == []:
                temps_str.append(r"$T$ = " + str(temp) + " K")
            else:
                temps_str.append(str(temp) + " K")
            # プロット:raw
            xs_raw.append(df["B_raw"])
            Rxx_raw.append(df["Rxx_raw"])
            Ryx_raw.append(df["Ryx_raw"])
            Gxx_raw.append(df["Rxx_raw"] / (df["Rxx_raw"] ** 2 + df["Ryx_raw"] ** 2))
            Gxy_raw.append(df["Ryx_raw"] / (df["Rxx_raw"] ** 2 + df["Ryx_raw"] ** 2))
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
            sub_x_step=2,
            y_step=3,
            sub_y_step=2,
            savepath="",
        )
        # Arrott plot
        igor.plot(
            ax=ax[3][3],
            xs=BoverRyx,
            ys=Ryx_d_sq,
            xlabel=r"$B / \rho_{yx}$ (a.u.)",
            ylabel=r"$\rho_{yx}^2$ (a.u.)",
            labels=[""] * len(BoverRyx),
            scale_x=1,
            scale_y=1,
            x_step=3,
            sub_x_step=2,
            y_step=3,
            sub_y_step=2,
            savepath=savepath,
        )

        return fig

    def plot_summary_data(self, df_summary, savepath=""):
        print("==================plot summary=========================")
        # グラフ描画準備
        igor = Igor(digits=2)
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 12))
        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.3)
        # fig.text(0.1, 0.9, sample_str)

        # プロット:temp-dep
        # ax[0].plot(
        #     df_summary["temps"], df_summary["carrier2D_abs"], marker="o", color="red"
        # )
        # ax[0].set_ylabel("carrier 2D(cm-2)")
        if "carrier2D_abs" in df_summary.columns:
            igor.plot(
                ax=ax[0],
                xs=[df_summary["temps"].values],
                ys=[df_summary["carrier2D_abs"].values],
                xlabel=r"",
                ylabel=r"carrier 2D($\mathrm{cm^{-2}}$)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick="x",
                title="",
                suffix="",
                savepath="",
            )
        elif "carrier3D_abs" in df_summary.columns:
            # ax[1].plot(
            #     df_summary["temps"], df_summary["carrier3D_abs"], marker="o", color="red"
            # )
            # ax[1].set_ylabel(r"carrier 3D($cm^{-3}$)")
            igor.plot(
                ax=ax[1],
                xs=[df_summary["temps"].values],
                ys=[df_summary["carrier3D_abs"].values],
                xlabel=r"",
                ylabel=r"carrier 3D($\mathrm{cm^{-3}}$)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick="x",
                title="",
                suffix="",
                savepath="",
            )
        if "mobility" in df_summary.columns:
            # ax[2].plot(df_summary["temps"], df_summary["mobility"], marker="o", color="red")
            # ax[2].set_ylabel(r"mobility ($cm^2 V^{-1} s{-1}$)")
            igor.plot(
                ax=ax[1],
                xs=[df_summary["temps"].values],
                ys=[df_summary["mobility"].values],
                xlabel=r"",
                ylabel=r"mobility ($\mathrm{cm^2 V^{-1} s^{-1}}$)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick="x",
                title="",
                suffix="",
                savepath=savepath,
            )
        if "RyxA" in df_summary.columns:
            igor.plot(
                ax=ax[2],
                xs=[df_summary["temps"].values],
                ys=[df_summary["RyxA"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$\rho_{yx}$A (Ω)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                log_scale="",
                grid="",
                omit_tick="",
                title="",
                suffix="",
                savepath=savepath,
            )
        else:
            pass
        # ax[2].set_xlabel(r"$T$ (K)")
        return fig

    # ################################  Library  ########################################
    def split_up_down(self, x_raw, y_raw):
        """
        データをup sweepとdown sweepに分ける
        x_raw: x_raw of sweep
        y_raw: y_raw of sweep
        """
        size = len(x_raw)
        is_sweep_up_down = False if x_raw[0] > x_raw[size // 4] else True
        # print("up to down" if(is_sweep_up_down) else "down to up")
        if is_sweep_up_down:
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

    def common_part(self, x, y):
        # 1D-arrayXとYの範囲の共通部分に属する要素のみを残す
        # 共通部分の最大値＝それぞれのarrayの最大値の中で最小のもの
        max_value = np.min([np.max(x), np.max(y)])
        min_value = np.max([np.min(x), np.min(y)])
        concat_array = np.concatenate([x, y])
        common_array = sorted(set([x for x in concat_array if x <= max_value and x >= min_value]))
        return common_array

    def symmetrize(self, x_raw_u, y_raw_u, x_raw_d=None, y_raw_d=None):
        """
        データを折り返し対称にする
        x_raw_u: x_raw of up sweep
        y_raw_u: y_raw of up sweep
        x_raw_d: x_raw of down sweep
        y_raw_d: y_raw of down sweep

        """
        if x_raw_d is None and y_raw_d is None:  # もしdown sweepがないなら
            y_int = interpolate.interp1d(x_raw_u, y_raw_u)
            x_ref = np.array([x for x in x_raw_u if x <= np.max(x_raw_u * -1) and x >= np.min(x_raw_u * -1)])
            y_sym = (y_int(x_ref) + y_int(-1 * x_ref)) / 2
            y_asym = (y_int(x_ref) - y_int(-1 * x_ref)) / 2
            return (x_ref, y_sym, y_asym)
        else:  # もしdown sweepがあるなら
            y_int_u = interpolate.interp1d(x_raw_u, y_raw_u)
            y_int_d = interpolate.interp1d(x_raw_d, y_raw_d)
            x_ref_u = self.common_part(x_raw_d, x_raw_u)
            x_ref_d = x_ref_u[::-1]
            y_sym_d = (y_int_d(x_ref_d) + y_int_u(x_ref_u)) / 2
            y_asym_d = (y_int_d(x_ref_d) - y_int_u(x_ref_u)) / 2
            # y_sym_u = y_sym_d[::-1]
            y_sym_u = y_sym_d
            # y_asym_u = y_asym_d[::-1]
            y_asym_u = y_asym_d * -1
            return (x_ref_u, y_sym_u, y_asym_u, x_ref_d, y_sym_d, y_asym_d)

    def diff_up_down_sweep(self, B_u, R_u, B_d, R_d):
        """NOTE: 現在使用していない

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
        B_ref = self.common_part(B_d, B_u)
        R_diff = R_u_int(B_ref) - R_d_int(B_ref)
        R_diff_abs = [(R_u_int(x) - R_d_int(x)) if x > 0 else (R_d_int(x) - R_u_int(x)) for x in B_ref]
        return B_ref, R_diff, R_diff_abs


class PPMS_RT:
    def __init__(self, filepath, ch_num, sample, thickness=0):
        self.filepath = filepath
        self.ch_num = ch_num
        self.sample = sample
        self.thickness = thickness

    def analyze_RT_ppms(self):
        # 保存ファイルパスを決定
        self.filename, fileext, self.savepath = make_savepath(self.filepath, savedir=self.sample)
        df = pd.read_csv(filepath, skiprows=range(0, 31))
        # ファイル取得 dat, ch_numに対応する列を取得
        B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx = get_data_columns_from_ppms_file(self.filepath, self.ch_num)
        df_RT = pd.DataFrame(np.array([temp, B, Rxx, Ryx, I_Rxx, I_Ryx]).T)
        df_RT.columns = ["temp", "B", "Rxx", "Ryx", "I_Rxx", "I_Ryx"]
        # 3Dの抵抗を表記
        if self.thickness != 0:
            df_RT["Rxx_3D"] = df_RT["Rxx"] * self.thickness * 1e2
            df_RT["Ryx_3D"] = df_RT["Ryx"] * self.thickness * 1e2
        # ファイル出力
        df_RT.to_csv(os.path.join(self.savepath, self.filename + ".csv"), index=False)
        # グラフ描画
        _ = self.plot_RT_data(
            df_RT,
            savepath=os.path.join(self.savepath, self.filename + ".png"),
        )
        plt.show()
        return None

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
                scale_x=1,
                scale_y=1e-3,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
            )
            igor.plot(
                ax=ax[0][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Ryx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{yx}$2D (kΩ)",
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Ryx"].max() < 0 else 0,
                max_y=None,
                scale_x=1,
                scale_y=1e-3,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
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
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
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
                max_y=None,
                scale_x=1,
                scale_y=1,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
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
                max_y=None,
                scale_x=1,
                scale_y=1e6,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
            )
            igor.plot(
                ax=ax[2][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Ryx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{yx}$ (μA)",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1e6,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath=self.savepath,
            )
        else:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            plt.subplots_adjust(wspace=0.5, hspace=0.2)
            # print("-----------plot Rxx vs T-----------")
            igor.plot(
                ax=ax[0][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Rxx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{xx}$2D (kΩ)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Rxx"].max() < 0 else 0,
                max_y=None,
                scale_x=1,
                scale_y=1e-3,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
            )
            # print("-----------plot Ryx vs T-----------")
            igor.plot(
                ax=ax[0][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["Ryx"].values],
                xlabel=r"",
                ylabel=r"$\rho_{yx}$2D (kΩ)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None if df_RT["Ryx"].max() < 0 else 0,
                max_y=None,
                scale_x=1,
                scale_y=1e-3,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
            )
            # print("-----------plot I_Rxx vs T-----------")
            igor.plot(
                ax=ax[1][0],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Rxx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{xx}$ (μA)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1e6,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                savepath="",
            )
            igor.plot(
                ax=ax[1][1],
                xs=[df_RT["temp"].values],
                ys=[df_RT["I_Ryx"].values],
                xlabel=r"$T$ (K)",
                ylabel=r"$I$ thru $\rho_{yx}$ (μA)",
                labels=[""],
                label_pos="best",
                min_x=0,
                max_x=300,
                min_y=None,
                max_y=None,
                scale_x=1,
                scale_y=1e6,
                x_step=4,
                sub_x_step=2,
                y_step=6,
                sub_y_step=2,
                suffix="",
                savepath=savepath,
            )

        return fig

    ###########################


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
            # print("makedirs : " + savepath)
            os.makedirs(savepath)
    return filename, fileext, savepath


# def get_data_columns_from_ppms_file(filepath, ch_num):
#     # ファイル取得 dat, ch_numに対応する列を取得
#     df = pd.read_csv(filepath, skiprows=range(0, 31))
#     B = df["Magnetic Field (Oe)"] * 1e-4
#     temp = df["Temperature (K)"]
#     Rxx = df["Bridge " + str(ch_num[0]) + " Resistivity (Ohm-cm)"]
#     Ryx = df["Bridge " + str(ch_num[1]) + " Resistivity (Ohm-cm)"]
#     I_Rxx = df["Bridge " + str(ch_num[0]) + " Excitation (uA)"] * 1e-6
#     I_Ryx = df["Bridge " + str(ch_num[1]) + " Excitation (uA)"] * 1e-6
#     return B, temp, Rxx, Ryx, I_Rxx, I_Ryx


def get_arrays_from_ppms_df(df, ch_num):
    """各々に対応するdat, ch_numに対応する列を取得

    Args:
        df (pd.DataFrame): datファイルを変換したcsvファイルから作成したDataFrame
        ch_num (List[str]): 測定に使用したチャンネル番号

    Returns:
        _type_: _description_
    """
    B = df["Magnetic Field (Oe)"].values * 1e-4
    temp = df["Temperature (K)"].values
    theta = df["Sample Position (deg)"].values
    Rxx = df["Bridge " + str(ch_num[0]) + " Resistivity (Ohm-cm)"].values
    Ryx = df["Bridge " + str(ch_num[1]) + " Resistivity (Ohm-cm)"].values
    I_Rxx = df["Bridge " + str(ch_num[0]) + " Excitation (uA)"].values * 1e-6
    I_Ryx = df["Bridge " + str(ch_num[1]) + " Excitation (uA)"].values * 1e-6
    return B, temp, theta, Rxx, Ryx, I_Rxx, I_Ryx


def get_data_columns_from_ppms_file_rot(filepath, ch_num):
    # ファイル取得 dat, ch_numに対応する列を取得
    df = pd.read_csv(filepath, skiprows=range(0, 31))
    B = df["Magnetic Field (Oe)"] * 1e-4
    temp = df["Temperature (K)"]
    sample_position = df["Sample Position (deg)"]
    Rxx = df["Bridge " + str(ch_num[0]) + " Resistivity (Ohm-cm)"]
    Ryx = df["Bridge " + str(ch_num[1]) + " Resistivity (Ohm-cm)"]
    I_Rxx = df["Bridge " + str(ch_num[0]) + " Excitation (uA)"] * 1e-6
    I_Ryx = df["Bridge " + str(ch_num[1]) + " Excitation (uA)"] * 1e-6

    return B, temp, sample_position, Rxx, Ryx, I_Rxx, I_Ryx


def smooth_data(self, x_raw, convolve_size=41):
    """データを平滑化する
    NOTE: 現在使用していない

    Args:
        x_raw (_type_): _description_
        convolve_size (int, optional): _description_. Defaults to 41.

    Returns:
        _type_: _description_
    """
    # smooth
    convolve_array = np.ones(convolve_size) / convolve_size
    x_raw_convolve = np.convolve(x_raw, convolve_array, mode="valid")
    return x_raw_convolve
