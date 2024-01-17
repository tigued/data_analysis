# 吉見さんのxrd_rigakuを整理したもの
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import interpolate
from scipy import optimize
from scipy.fftpack import fft, fftfreq
from scipy import signal
from typing import List, Tuple
import pathlib
from pathlib import Path
import argparse

from igor import Igor  # , round_down, round_up

# use these lines on top of your matplotlib script
import matplotlib.ticker


def main(sample):
    directory = Path("/Users/uedataiga/Desktop/grad-research/data_analysis/Data/XRD/")
    filepath_2tw = directory / f"{sample}/2tw.dat"
    filepath_r006 = directory / f"{sample}/rock006.dat"
    filepath_r0015 = directory / f"{sample}/rock15.dat"
    xrd = XRD(sample, filepath_2tw, filepath_r006, filepath_r0015)
    xrd.run()
    # xrd.export_igor_command()


def main_summary(samples, labels=None, offsets_2tw=None, summary_savepath=None):
    result_diranme = ""
    for sample in samples:
        result_diranme += f"{sample}_"
    result_diranme = result_diranme[:-1]

    if labels is None:
        labels = samples
    if offsets_2tw is None:
        offsets_2tw = [10**i for i in range(len(samples))]
    offsets_2tw = offsets_2tw[::-1]
    xrd_summary = {
        "sample": samples,
        "x_2tw": [],
        "y_2tw": [],
        "x_006": [],
        "y_006": [],
        "x_0015": [],
        "y_0015": [],
    }
    directory = Path("/Users/uedataiga/Desktop/grad-research/data_analysis/Data/XRD/")
    save_dir = Path(f"/Users/uedataiga/Desktop/grad-research/data_analysis/Data/summary/{result_diranme}")
    if not save_dir.exists():
        save_dir.mkdir()

    for sample, offset in zip(samples, offsets_2tw):
        csv_2tw = directory / f"{sample}/2tw.csv"
        # csv_r006 = directory / f"{sample}/rock006.csv"
        # csv_r0015 = directory / f"{sample}/rock15.csv"
        df_2tw = pd.read_csv(csv_2tw)
        df_2tw["int"] *= offset
        # df_2tw_fft = pd.read_csv(csv_2tw_fft)
        # df_r006 = pd.read_csv(csv_r006)
        # df_r0015 = pd.read_csv(csv_r0015)
        xrd_summary["x_2tw"].append(df_2tw["2theta"].values)
        xrd_summary["y_2tw"].append(df_2tw["int"].values)
        # xrd_summary["x_006"].append(df_r006["d_omega"].values)
        # xrd_summary["y_006"].append(df_r006["int"].values)
        # xrd_summary["x_0015"].append(df_r0015["d_omega"].values)
        # xrd_summary["y_0015"].append(df_r0015["int"].values)
    # 2tw
    suffix = ""
    for sample in samples:
        suffix += f"_{sample}"
    igor = Igor(digits=1, fontsize=25)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    igor.plot(
        ax=ax,
        xs=xrd_summary["x_2tw"],
        ys=xrd_summary["y_2tw"],
        xlabel=r"$\mathrm{2\theta}$ (deg)",
        ylabel="Intensity (cps)",
        labels=labels,
        label_pos=(1, 0.9),  # "best",
        min_x=0,
        max_x=80,
        x_step=6,
        sub_x_step=2,
        min_y=1,
        max_y=1e16,
        y_step=5,
        sub_y_step=2,
        log_scale="y",
        grid="",
        title="",
        suffix=suffix,
        savepath=str(save_dir / "2tw_summary.png"),
    )
    igor = Igor(digits=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    igor.plot(
        ax=ax,
        xs=xrd_summary["x_2tw"],
        ys=xrd_summary["y_2tw"],
        xlabel=r"$\mathrm{2\theta}$ (deg)",
        ylabel="Intensity (cps)",
        labels=labels,
        label_pos=(1, 0.9),  # "best",
        min_x=12,
        max_x=22,
        x_step=6,
        sub_x_step=2,
        min_y=1,
        max_y=1e16,
        y_step=5,
        sub_y_step=2,
        log_scale="y",
        grid="",
        title="near (006)",
        suffix=suffix,
        savepath=str(save_dir / "2tw_summary_near(006).png"),
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    igor.plot(
        ax=ax,
        xs=xrd_summary["x_2tw"],
        ys=xrd_summary["y_2tw"],
        xlabel=r"$\mathrm{2\theta}$ (deg)",
        ylabel="Intensity (cps)",
        labels=labels,
        label_pos=(1, 0.9),  # "best",
        min_x=40,
        max_x=50,
        x_step=6,
        sub_x_step=2,
        min_y=1,
        max_y=1e16,
        y_step=5,
        sub_y_step=2,
        log_scale="y",
        grid="",
        title="near (0015)",
        suffix=suffix,
        savepath=str(save_dir / "2tw_summary_near(0015).png"),
    )
    plt.show()

    igor = Igor(digits=2)
    fig = plt.figure(figsize=(6, 6))
    emptylabels = [""] * len(labels)
    ax1 = fig.add_subplot(1, 2, 1)
    igor.plot(
        ax=ax1,
        xs=xrd_summary["x_006"],
        ys=xrd_summary["y_006"],
        xlabel=r"$\Delta \omega$",
        ylabel=r"Intensity (cps)",
        labels=emptylabels,
        label_pos="upper left",
        min_x=-1,
        max_x=1,
        x_step=5,
        sub_x_step=2,
        min_y=0,
        max_y=10000,
        y_step=5,
        sub_y_step=2,
        log_scale="",
        grid="",
        title="",
        suffix=suffix,
        savepath="",
    )
    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)

    igor.plot(
        ax=ax2,
        xs=xrd_summary["x_0015"],
        ys=xrd_summary["y_0015"],
        xlabel=r"$\Delta \omega$",
        ylabel=r"",
        labels=emptylabels,
        label_pos="best",
        min_x=-1,
        max_x=1,
        x_step=5,
        sub_x_step=2,
        min_y=0,
        max_y=10000,
        y_step=5,
        sub_y_step=2,
        log_scale="",
        grid="",
        omit_tick="y",
        title="",
        suffix=suffix,
        savepath=str(save_dir / "rocking_summary.png"),
    )
    plt.show()

    return xrd_summary


class XRD:
    def __init__(self, sample, filepath_2tw="", filepath_r006="", filepath_r0015=""):
        self.SAMPLE = sample
        self.filepath_2tw = filepath_2tw
        self.filepath_r006 = filepath_r006
        self.center = None
        self.fwhm = None
        self.filepath_r0015 = filepath_r0015
        self.filedir = Path(os.path.dirname(filepath_2tw))
        try:
            self.df = pd.read_csv(self.filepath_2tw, skiprows=2, delimiter="\\s")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.filepath_2tw, skiprows=2, delimiter="\\s", encoding="shift-jis")
        # 低角反射の設定
        self.two_theta_min = 0.5
        self.two_theta_max = 5
        self.t_upperbound = 200  # nm

    def run(self):
        print("##### two_theta_omega #####")
        csv_2tw = self.two_theta_omega()
        print("##### extract_low_angle #####")
        _ = self.extract_low_angle()
        print("##### detect_thickness #####")
        csv_2tw_fft = self.detect_thickness()
        print("##### laue_fringe #####")
        self.laue_fringe()
        print("##### rocking #####")
        csv_r006 = self.rocking(self.filepath_r006, title="(006) rocking")
        csv_r0015 = self.rocking(self.filepath_r0015, title="(0015) rocking")
        # self.integrate_data(
        #     [csv_2tw, csv_2tw_fft, csv_r006, csv_r0015],
        #     prefix_list=[
        #         f"_{self.SAMPLE}_2tw",
        #         f"_{self.SAMPLE}_2tw_fft",
        #         f"_{self.SAMPLE}_r006",
        #         f"_{self.SAMPLE}_r0015",
        #     ],
        # )
        return csv_2tw, csv_2tw_fft

    # def integrate_data(self, path_list, prefix_list):
    #     """Igorの読み込みを楽にするためにすべてのcsvを一つに結合する

    #     Args:
    #         path_listmprefix_list (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     df_master = pd.DataFrame([])
    #     for path, prefix in zip(path_list, prefix_list):
    #         try:
    #             tmp = pd.read_csv(path, index_col=None)
    #         except UnicodeDecodeError:
    #             tmp = pd.read_csv(path, index_col=None, encoding="shift-jis")
    #         tmp = tmp.rename(columns={col: col + prefix for col in tmp.columns})
    #         df_master = pd.concat([df_master, tmp], axis=1)
    #     df_master.to_csv(self.filedir / "df_master.csv", index=False)
    #     return None

    def two_theta_omega(self) -> Tuple[pd.DataFrame, pathlib.PosixPath]:
        xrd_lambda = 1.540598  # Kα（カッパアルファ1）の波長
        filename, _ = os.path.splitext(os.path.basename(self.filepath_2tw))

        self.df.columns = ["2theta", "int"]
        self.df["int"] += 1.0
        csv_path = self.filedir / f"X{filename}.csv"  # NOTE: +1したものであることに注意
        self.df.to_csv(csv_path, index=False)

        self.df["q"] = 4 * np.pi * np.sin(self.df["2theta"] * np.pi / 360) / xrd_lambda  # q = 4πsin(θ)/λ
        # qは散乱ベクトルの大きさを表す
        self.df["q_lnsp"] = np.linspace(np.min(self.df["q"]), np.max(self.df["q"]), np.size(self.df["q"]))  # qの等間隔な配列
        lar_intp = interpolate.interp1d(self.df["q"], self.df["int"])  # qの等間隔な配列に対応するように補間
        self.df["int_log"] = np.log10(lar_intp(self.df["q_lnsp"]) + 1)  # 補間した値を対数変換

        min_x, max_x = 0, 80
        min_y, max_y = 1e0, 1e10
        x_step, y_step = 5, 6
        sub_x_step, sub_y_step = 2, 2
        igor = Igor(fontsize=28)
        fig, ax = plt.subplots(figsize=(15, 6))
        igor.plot(
            ax=ax,
            xs=[self.df["2theta"]],
            ys=[self.df["int"]],
            xlabel=r"$ \mathrm{2\theta} $ (deg)",
            ylabel=r"Inteisity (cps)",
            labels=[""],
            label_pos="best",
            min_x=min_x,
            max_x=max_x,
            x_step=x_step,
            sub_x_step=sub_x_step,
            min_y=min_y,
            max_y=max_y,
            y_step=y_step,
            sub_y_step=sub_y_step,
            log_scale="y",
            grid="",
            title="",
            suffix="",
            savepath=str(self.filedir / "2theta-omega.png"),
        )
        plt.show()
        return csv_path

    def extract_low_angle(self) -> List[float]:
        # 低角反射（two_theta_min~two_theta_max）で膜厚を求める

        self.df_trim = self.df.loc[self.df.index[(self.df["2theta"] > self.two_theta_min) & (self.df["2theta"] < self.two_theta_max)].tolist()]  # 低角反射のみ抽出

        ###############################################
        def xrd_ref_bg(q, a, b, c):  # XRD強度の背景
            return -a * np.log10((q - b)) + c

        ###############################################

        param_a = np.linspace(0.1, 500, 51)
        param_b = np.linspace(-10, 0, 16)
        param_c = np.linspace(-5, 5, 11)

        for a, b, c in itertools.product(param_a, param_b, param_c):
            # print(a, b, c)
            param_init = [a, b, c]
            try:
                param_opti, _ = optimize.curve_fit(
                    xrd_ref_bg,
                    self.df_trim["q_lnsp"].values,
                    self.df_trim["int_log"].values,
                    p0=param_init,
                )  # 最適化
            except RuntimeError as e:
                print(f"cannot optimize: {e}")
                continue
            if np.isnan(param_opti).any():
                # print("cannot optimize")
                continue
            print(param_opti)
            self.df_trim["int_bg_log"] = xrd_ref_bg(self.df_trim.loc[:, "q_lnsp"], *param_opti)  # XRD強度の背景
            self.df_trim["int_bg"] = 10 ** self.df_trim["int_bg_log"]
            self.df_trim["int_osci"] = self.df_trim["int_log"] - self.df_trim["int_bg_log"]
            break

        # min_x, max_x = 0, 0.35
        # # min_y, max_y = 1e0, 1e10
        # min_y, max_y = 0, 10

        igor = Igor(digits=2)
        fig = plt.figure(figsize=(5, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        igor.plot(
            ax=ax1,
            xs=[self.df_trim["q_lnsp"], self.df_trim["q_lnsp"]],
            ys=[self.df_trim["int_bg_log"], self.df_trim["int_log"]],
            labels=["", ""],
            xlabel=r"",
            ylabel=r"Intensity(log) (cps)",
            min_y=0,
            max_y=10,
            y_step=5,
            sub_y_step=2,
            omit_tick="x",
            savepath="",
        )

        # min_x, max_x = 0, 0.35
        # min_y = round_down(1.1 * min(self.df_trim["int_osci"]), 2)
        # max_y = round_up(
        #     1.1 * max(self.df_trim["int_osci"]), 2
        # )
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        igor.plot(
            ax=ax2,
            xs=[self.df_trim["q_lnsp"]],
            ys=[self.df_trim["int_osci"]],
            xlabel=r"q ($\mathrm{nm}^{-1}$)",  # 散乱ベクトルの大きさ # r"q (nm$^{-1}$)" = 4πsin(θ)/λ
            ylabel=r"difference from background (cps)",
            label=[""],
            label_pos="best",
            min_x=0,
            max_x=0.3,
            x_step=4,
            sub_x_step=2,
            min_y=None,
            max_y=None,
            y_step=5,
            sub_y_step=2,
            grid="",
            omit_tick="",
            title="",
            suffix="",
            savepath=str(self.filedir / "2tw_low_angle.png"),
        )
        plt.show()
        return param_opti

    def laue_fringe(self, two_theta_min=14, two_theta_max=23):
        """参考：CBSTの場合
                (006): 14 < 2θ < 23にありがち
                (0015:)
        :
                Args:
                    two_theta_min (int, optional): _description_. Defaults to 14.
                    two_theta_max (int, optional): _description_. Defaults to 23.

                Returns:
                    _type_: _description_
        """
        # NOTE: 未完成．006と0015でどちらもやる．目標は膜圧求めるところまで
        # ラウエフリンジの解析（各レイヤー層での干渉．(sinN/N)^2(0次の場合)のフィッティング）
        two_theta_min = 14  # (0015)のピークが入る範囲を勘で定める
        two_theta_max = 23
        thick_init = 40  # nm

        df_trim_laue = self.df.loc[self.df.index[(self.df["2theta"] > two_theta_min) & (self.df["2theta"] < two_theta_max)].tolist()]

        # 初期値推定
        # 2theta_omegaの最大値からd_initを推定
        # two_theta_peak = df_trim_laue["2theta"].values[
        #     np.argmax(df_trim_laue["int_log"])
        # ]
        q_peak = df_trim_laue["q"].values[np.argmax(df_trim_laue["int_log"])]
        d_init = 2 * np.pi / q_peak
        M_init = thick_init / d_init

        def laue_func(q, d, M, A):
            return np.log(np.sin(q * d * M / 2) ** 2) - np.log(np.sin(q * d / 2) ** 2) + A

        # NOTE: 現状うまくフィッティングできない
        # param_init = [M_init, 10]
        # param_opti, _ = optimize.curve_fit(lambda q, M, A: laue_func(q, d_init, M, A), df_trim["q_lnsp"].values, df_trim["int_log"].values, p0 = param_init)
        # plt.plot(df_trim["q"], laue_func(df_trim["q"].values, d_init, *param_opti))

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1)
        igor = Igor(digits=2, margin=0.1)
        igor.plot(
            ax=ax,
            xs=[
                df_trim_laue["q"],
                # df_trim_laue["q"]
            ],
            ys=[
                df_trim_laue["int_log"],
                # laue_func(df_trim_laue["q"].values, d_init, M_init, 3),
            ],
            xlabel=r"q ($\mathrm{nm}^{-1}$)",
            ylabel=r"Intensity(log) (cps)",
            # labels=["", ""],
            labels=[""],
            label_pos="best",
            min_x=None,
            max_x=None,
            x_step=5,
            sub_x_step=2,
            min_y=0,
            max_y=None,
            y_step=5,
            sub_y_step=2,
            grid="",
            title="",
            suffix="",
            savepath=str(self.filedir / "laue_fringe.png"),
        )
        plt.show()

    def detect_thickness(self):
        filename, _ = os.path.splitext(os.path.basename(self.filepath_2tw))
        int_fft = fft(self.df_trim["int_osci"].values)  # 低角反射（補正済み）をフーリエ変換
        N = len(self.df_trim["int_osci"])
        q_spacing = np.abs(self.df_trim["q_lnsp"].values[1] - self.df_trim["q_lnsp"].values[0])  # qの間隔
        self.df_trim["q_freq"] = fftfreq(N, q_spacing)  # qの周波数

        # パワースペクトラムから膜厚を求める
        # ps = power spectrum
        self.df_trim["int_ps"] = np.abs(int_fft)
        # dq = period of oscillation
        # q_freq = inverse of dq
        self.df_trim["q_period"] = 1 / self.df_trim["q_freq"]
        self.df_trim["thickness"] = 2 * np.pi * self.df_trim["q_freq"] * 0.1  # 0.1はAからnmへの変換
        csv_path = self.filedir / f"X{filename}_fft.csv"
        self.df_trim.to_csv(csv_path, index=False)

        igor = Igor(digits=2)
        fig, ax = plt.subplots(figsize=(5, 4))
        min_x, max_x = 0, self.t_upperbound
        min_y, max_y = 0, None
        igor.plot(
            ax=ax,
            xs=[self.df_trim["thickness"]],
            ys=[self.df_trim["int_ps"]],
            xlabel=r"thickness (nm)",
            ylabel=r"power spectrum",
            label=[""],
            label_pos="best",
            # fmt="o-",
            min_x=min_x,
            max_x=max_x,
            x_step=5,
            sub_x_step=2,
            min_y=min_y,
            max_y=max_y,
            y_step=5,
            sub_y_step=2,
            log_scale="",  # "y",
            grid="",
            title="",
            suffix="",
            savepath=str(self.filedir / "thickness.png"),
        )
        plt.show()

        # ### search preak ####
        fft_peak = signal.argrelmax(self.df_trim["int_ps"].values)
        fft_peak_value = self.df_trim["int_ps"].values[fft_peak]
        q_osci = 1 / self.df_trim["q_freq"].values[fft_peak]
        thickness_nm = 2 * np.pi * 0.1 * self.df_trim["q_freq"].values[fft_peak]
        fft_result = np.array([fft_peak_value, q_osci, thickness_nm])
        fft_result = fft_result[:, fft_result[0, :].argsort()[::-1]]
        fft_peak_value = fft_result[0]
        q_osci = fft_result[1]
        thickness_nm = fft_result[2]

        for i, x in enumerate(fft_peak_value):
            if i == 0:
                fft_result = ""
            if thickness_nm[i] > 0 and thickness_nm[i] < self.t_upperbound and x > fft_peak_value[0] * 0.2:
                fft_result += "period of q_oscillation = {:.3f}".format(q_osci[i])
                fft_result += "\nthickness = {:.2f} nm".format(thickness_nm[i])
                fft_result += "\nps = {0:.1f}".format(x)
                fft_result += "\n\n"
        print(fft_result)
        fft_result_path = self.filedir / f"{filename}_fft_summary.txt"
        with open(fft_result_path, mode="w") as f:
            f.write(fft_result)

        return csv_path

    def detect_c_axis_length(self, theta0=0, target="006"):
        """(006) or (0015)中心のthetaをθ_0，c軸長をd，x線の波長をλとすると，以下の式が成り立つ．
        d = λ / (2 * sin(θ_0 / 2))
        この式を用いてc軸長を計算する．

        Args:
            data (Dict): _description_
        """
        print(f"target = {target}")
        # c_axis_length = 0
        xrd_lambda = 1.540598  # Kα（カッパアルファ1）の波長（Å)
        d = (3 / 5) * (xrd_lambda / (2 * np.sin(np.deg2rad(theta0))))
        if target == "006":
            d = d
        elif target == "0015":
            d = d * (5 / 2)
        return d

    def rocking(self, filepath, title="rocking curve"):
        """_description_

        Args:
            filename (str): ファイル名

        Returns:
            _type_: _description_
        """
        # filedir = os.path.dirname(filepath)
        # filebasename = os.path.basename(filepath)
        # filename, fileext = os.path.splitext(filebasename)
        filename, _ = os.path.splitext(os.path.basename(filepath))
        try:
            df = pd.read_csv(filepath, skiprows=2, delimiter=" ")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, skiprows=2, delimiter=" ", encoding="shift-jis")
        df.columns = ["omega", "int"]
        # plt.plot(df["omega"], df["int"], ".", color="red")

        ###############################################
        def lor(x, a, b, c):
            return c / ((x - a) ** 2 + b)

        ###############################################

        # 初期値設定
        int_max = np.max(df["int"].values)
        omega_max = df["omega"].values[np.argmax(df["int"].values)]
        a = omega_max
        b = 1 / 400  # 半値幅0.1くらい
        c = int_max * b
        # fitting
        param_init = [a, b, c]
        param_opti, _ = optimize.curve_fit(lor, df["omega"].values, df["int"].values, p0=param_init, maxfev=1000)


        # dfにフィッティング結果を詰める
        df["d_omega"] = df["omega"] - param_opti[0]
        df["lor_fit"] = lor(df["omega"], *param_opti)
        # print(df)
        center = round(param_opti[0], 3)
        fwhm = round(2 * np.sqrt(param_opti[1]), 3)
        if "006" in filename:
            csv_path = self.filedir / "rock006.csv"
            print(f"center: {center}\nFWHM: {fwhm}")
            title = "(006) rocking"
            c_axis_length = self.detect_c_axis_length(theta0=param_opti[0], target="006")
        elif "15" in filename:
            csv_path = self.filedir / "rock0015.csv"
            title = "(0015) rocking"
            print(f"center: {center}\nFWHM: {fwhm}")
            c_axis_length = self.detect_c_axis_length(theta0=param_opti[0], target="0015")

        else:
            print("other peak is detected")
            csv_path = self.filedir / f"rock{filename}.csv"
            title = f"({filename}) rocking"

        fit_result = "center: {0:.3f}\nFWHM: {1:.3f}\nc_axis_length: {2: .3f}".format(param_opti[0], 2 * np.sqrt(param_opti[1]), c_axis_length)
        print(fit_result)
        df.to_csv(csv_path, index=False)

        igor = Igor(digits=2)
        fig, ax = plt.subplots(figsize=(3, 5))
        min_x, max_x = -1, 1
        min_y = 0
        if np.max(df["int"].values) > 10000:
            max_y = None
        else:
            max_y = 10000

        igor.plot(
            ax=ax,
            xs=[df["d_omega"].values],
            ys=[df["int"].values],
            xlabel=r"$\Delta \omega$",
            ylabel=r"Intensity (cps)",
            label=[""],
            label_pos="best",
            min_x=min_x,
            max_x=max_x,
            x_step=5,
            sub_x_step=2,
            min_y=min_y,
            max_y=max_y,
            y_step=5,
            sub_y_step=2,
            grid="",
            title=title,
            suffix="",
            savepath=str(self.filedir / f"{filename}.png"),
        )
        plt.show()

        # フィッティングパラメータを出力
        fit_result_path = self.filedir / f"{filename}_fit.txt"
        with open(fit_result_path, mode="w") as f:
            f.write(fit_result)
        return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", help="sample number")
    args = parser.parse_args()
    sample = args.sample
    main(sample)
