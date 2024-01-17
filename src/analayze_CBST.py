import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path
from igor import Igor
import glob
from detect_peak import detect_peak
from pprint import pprint
import sys


def main():
    dirnames = [
        "230525_#1-1354",
        "230719_#1-1359_#1-1360",
        "231003_#1-1364_#1-1365",
        "231003_#1-1364_#1-1365",
        "230726_#1-1362",
        "231024_#1-1369_#1-1372",
        "231024_#1-1369_#1-1372",
        "231018_#1-1369_#1-1370",
        "230720_#1-1361_#2-1421",
        "230628_#1-1357(CBST)",
        # "",
    ]
    samples = [
        "#1-1354",
        "#1-1360",
        "#1-1364",
        "#1-1365",
        "#1-1362",
        "#1-1372",
        "#1-1369",
        "#1-1370",
        "#1-1361",
        "#1-1357",
        # "#1-1373"
    ]
    xlist = [0.0, 0.008, 0.012, 0.016, 0.02, 0.025, 0.03, 0.04, 0.05, 0.077]  # , 0.1]
    # dirnames = ["231030_#1-1374_#1-1375"]
    # samples = ["#1-1374"]  # , "#1-1375"]
    # xlist = [0.05]  # , 0.012, 0.016, 0.02, 0.025, 0.03, 0.04, 0.05, 0.077]  # , 0.1]
    ppms_dir = Path("/Users/uedataiga/Desktop/grad-research/data_analysis/Data/PPMS/")
    result_dirname = ""
    for sample in samples:
        result_dirname += f"{sample}_"
    result_dirname = result_dirname[:-1]
    savedir = Path(f"/Users/uedataiga/Desktop/grad-research/data_analysis/Data/summary/{result_dirname}/")
    if not savedir.exists():
        savedir.mkdir(parents=True)
    xrd_dir = Path("/Users/uedataiga/Desktop/grad-research/data_analysis/Data/XRD/")

    plot_TC_vs_x(samples, xlist, dirnames, ppms_dir, savedir)
    plot_histerisis_char_vs_x(samples, xlist, dirnames, ppms_dir, savedir)
    plot_HallAngle_vs_x(samples, xlist, dirnames, ppms_dir, savedir)
    # plot_transport_2K_vs_B(samples, xlist, dirnames, ppms_dir, savedir)

    # for sample in samples:
    #     plot_HC_vs_T(sample, dirnames, ppms_dir, savedir)

    # samplesとxlistから#1-1370を除外
    samples = np.delete(np.array(samples).copy(), 6)
    xlist = np.delete(np.array(xlist).copy(), 6)

    theta_0_006_list, theta_0_0015_list = plot_fwhm_vs_x(samples, xlist, dirnames, xrd_dir, savedir)
    plot_c_axis_length_vs_x(samples, xlist, theta_0_006_list, dirnames, xrd_dir, savedir)

    plot_growth_rate(
        samples,
        xlist,
        theta_0_006_list,
        xrd_dir,
        savedir / "growth_rate.png",
        epilog_path="/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx",
    )
    plot_carrier_dencity(samples, xlist, dirnames, ppms_dir, savedir)


def plot_transport_2K_vs_B(samples, xlist, dirnames, ppms_dir, savedir):
    xs, rhoxx, rhoyx, sigma_xx, sigma_xy = [], [], [], [], []
    for sample, dirname in zip(samples, dirnames):
        df = pd.read_csv(ppms_dir / dirname / "Hall_2.0K.csv")
        xs.append(pd.concat([df["B_ref_u"], df["B_ref_d"]]).values)
        rhoxx.append(pd.concat([df["Rxx_u"], df["Rxx_d"]]).values)
        rhoyx.append(pd.concat([df["Ryx_u"], df["Ryx_d"]]).values)
        sigma_xx.append(pd.concat([df["Gxx_u"], df["Gxx_d"]]).values)
        sigma_xy.append(pd.concat([df["Gxy_u"], df["Gxy_d"]]).values)

    igor = Igor(digits=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    igor.plot(
        ax=ax,
        xs=xs,
        ys=rhoxx,
        xlabel="$B$ (T)",
        ylabel="$\\rho_{xx}$ (lΩ)",
        labels=[""],
        min_x=-10,
        max_x=10,
        x_step=4,
        sub_x_step=2,
        min_y=0,
        max_y=1,
        y_step=4,
        sub_y_step=2,
        savepath=savedir / "rhoxx_vs_B.png",
        fmt="-",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    igor.plot(
        ax=ax,
        xs=xs,
        ys=rhoyx,
        xlabel="$B$ (T)",
        ylabel="$\\rho_{yx}$ (Ω)",
        labels=[""],
        min_x=-14,
        max_x=14,
        x_step=4,
        sub_x_step=2,
        min_y=0,
        max_y=1,
        y_step=4,
        sub_y_step=2,
        savepath=savedir / "rhoyx_vs_B.png",
        fmt="-",
    )
    fig, ax = plt.subplots(figsize=(5, 4))


def plot_carrier_dencity(samples, xlist, dirnames, ppms_dir, savedir):
    """T = 2 Kでのキャリア濃度をプロット

    Args:
        samples (_type_): _description_
        xlist (_type_): _description_
        dirnames (_type_): _description_
    """
    carrier_dencity_list = []
    for sample, dirname in zip(samples, dirnames):
        df = pd.read_csv(ppms_dir / dirname / "temp_dep.csv")
        idx_2K = np.argmin(np.abs(np.array(df["temps"]) - 2))
        c_d = float(df["carrier2D"][idx_2K])
        carrier_dencity_list.append(c_d)

    igor = Igor(digits=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    savepath = savedir / "carrier_dencity_vs_x_2K.png"
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[carrier_dencity_list],
        xlabel="x",
        ylabel="carrier dencity 2D (cm^-2)",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=1e21,
        y_step=6,
        sub_y_step=2,
        suffix="",
        savepath=savepath,
        fmt="o-",
    )


def plot_HC_vs_T(sample, dirnames, directory, savedir):
    """plot HC vs T
    #NOTE: 怪しい

    Args:
        sample (_type_): _description_
        dirnames (_type_): _description_
        directory (_type_): _description_
        savedir (_type_): _description_
    """
    igor = Igor(digits=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    savepath = savedir / "HC_vs_T.png"
    for dirname in dirnames:
        dirpath = directory / f"{dirname}/{sample[3:]}"
        df = pd.read_csv(dirpath / "temp_dep.csv")
        igor.plot(
            ax=ax,
            xs=[df["temps"].values],
            ys=[df["HC"].values],
            xlabel="$T$ (K)",
            ylabel="$H_C$ (Oe)",
            labels=[dirname],
            min_x=0,
            max_x=300,
            x_step=4,
            sub_x_step=2,
            min_y=0,
            max_y=1000,
            y_step=4,
            sub_y_step=2,
            grid="",
            suffix="",
            savepath=savepath,
            fmt="o-",
        )


def plot_TC_vs_x(samples: List[str], xlist: List[float], dirnames, directory, savedir):
    """
    Plot TC vs x for a list of samples
    """
    xlist = np.array(xlist).copy()
    TCs = []
    for idx, (dirname, sample) in enumerate(zip(dirnames, samples)):
        TC_path = directory / dirname / sample[3:] / f"TC.txt"
        with open(TC_path, "r") as f:
            # TC: 〇〇の形で入っている
            TC = f.readline()
            TC = TC.split(":")[1]
            TCs.append(float(TC))
    igor = Igor()
    fig, ax = plt.subplots(figsize=(5, 4))
    igor.plot(
        ax=ax,
        xs=[np.array(xlist)],
        ys=[np.array(TCs)],
        xlabel="x",
        ylabel=r"$T_C$ (K)",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=300,
        y_step=4,
        sub_y_step=2,
        grid="",
        savepath=savedir / "TC_vs_x.png",
        fmt="o-",
    )
    return TCs


def plot_histerisis_char_vs_x(samples, xlist, dirnames, directory, savedir):
    """ヒステリシスの特徴である保磁力と飽和抵抗率をCr濃度に対してプロット

    Args:
        samples (_type_): _description_
        xlist (_type_): _description_
        dirnames (_type_): _description_
        directory (_type_): _description_
        savedir (_type_): _description_

    Returns:
        _type_: _description_
    """
    xlist = np.array(xlist).copy()
    Hclist = []
    rho_xx_0T_list = []
    rho_yx_0T_list = []
    sigma_xy_0T_list = []
    sigma_xx_0T_list = []
    for i in range(len(samples)):
        sample = samples[i]
        dirname = dirnames[i]
        dirpath = directory / f"{dirname}/{sample[3:]}"

        globpath = glob.glob(str(dirpath / "Hall_2.*.csv"))
        df = pd.read_csv(globpath[0])
        # 欠損値除去
        df = df.dropna(how="any")

        # 保磁力H_c: Ryx_uの絶対値が0に最も近いときのB_ref_u
        # あるいは反対照化しているので同じだが，Ryx_dの絶対値が0に最も近いときのB_ref_dの絶対値
        # H_c = df["B_ref_u"][(np.abs(df["B_ref_u"]) <= 1.0) & (np.abs(df["Ryx_u"]).argmin())]
        try:
            df_tmp = df[(np.abs(df["B_ref_u"]) <= 1.0)]  # n型だと保磁力付近以外でもRyx_uが0になることがあるので，保磁力付近（1Tいない）のみを抽出
            H_c = df_tmp["B_ref_u"].iloc[np.abs(df_tmp["Ryx_u"]).argmin()]

            # 飽和抵抗率ρ_yx_0T: B_ref_dの絶対値が0に近いときのRyx_dの値
            # あるいは反対照化しているので同じだが，B_ref_uの絶対値が0に近いときのRyx_uの絶対値
            rho_xx_0T = df["Rxx_d"][np.abs(df["B_ref_d"]).argmin()]
            rho_yx_0T = df["Ryx_d"][np.abs(df["B_ref_d"]).argmin()]
            sigma_xy_0T = df["Gxy_d"][np.abs(df["B_ref_d"]).argmin()]
            sigma_xx_0T = df["Gxx_d"][np.abs(df["B_ref_d"]).argmin()]
        except KeyError:
            # BST（非磁性体の場合）

            H_c, rho_xx_0T, rho_yx_0T, sigma_xy_0T, sigma_xx_0T = 0.0, 7480.3, 0.0, 0.0, 0.0
        except ValueError as e:
            # 全てnanの場合
            print(e)
            print(f"no data in {sample}")
            # xlist.pop(i)
            xlist = np.delete(xlist, i)
            continue
        Hclist.append(H_c)
        rho_xx_0T_list.append(rho_xx_0T)
        rho_yx_0T_list.append(rho_yx_0T)
        sigma_xy_0T_list.append(sigma_xy_0T)
        sigma_xx_0T_list.append(sigma_xx_0T)

    print("sigma_xy_0T_list")
    pprint(sigma_xy_0T_list)
    print("sigma_xx_0T_list")
    pprint(sigma_xx_0T_list)
    print("rho_xx_list")
    pprint(rho_xx_0T_list)
    sys.exit()

    igor = Igor(digits=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    print("Hclist")
    pprint(Hclist)
    igor.plot(
        ax=ax,
        xs=[np.array(xlist)],
        ys=[np.array(Hclist)],
        xlabel="x",
        ylabel=r"$H_c$ (T)",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=1,
        y_step=6,
        sub_y_step=2,
        savepath=savedir / "Hc_vs_x.png",
        fmt="o-",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    print("rho_yx_0T_list")
    pprint(rho_yx_0T_list)
    igor.plot(
        ax=ax,
        xs=[np.array(xlist)],
        ys=[np.array(rho_yx_0T_list)],
        xlabel="x",
        ylabel=r"$\rho_{yx}$ @ $B$ = 0 T (kΩ)",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=None,
        max_y=None,
        scale_y=1e-3,
        savepath=savedir / "rho_yx_0T_vs_x.png",
        fmt="o-",
    )
    return Hclist, rho_yx_0T_list


def plot_HallAngle_vs_x(samples, xlist, dirnames, directory, savedir):
    hallangle_list = []
    xlist = np.array(xlist).copy()
    temps_list = []
    igor = Igor()
    for i in range(len(samples)):
        sample = samples[i]
        # i=0だけHall_2.0K.csvからホール角を求める
        dirpath = directory / f"{dirnames[i]}/{sample[3:]}"
        savepath_hallangle = dirpath / f"hall_angle_{sample}.png"
        # if i == 0:
        #     df = pd.read_csv(dirpath / "Hall_2.0K.csv")
        #     # hallangle = df["Ryx"] / df["Rxx"]
        #     hallangle = df["HallAngle_abs"]
        #     # plot(df["B_ref"], hallangle, label="", xlabel="$B$ (T)", ylabel="Hall Angle", title="", MIN_X=MIN_X, MAX_X=MAX_X, X_STEP=X_STEP, SUB_X_STEP=SUB_X_STEP, MIN_Y=MIN_Y, MAX_Y=MAX_Y, Y_STEP=Y_STEP, SUB_Y_STEP=SUB_Y_STEP, savepath=savepath_hallangle, hlines=False)
        #     fig, ax = plt.subplots(figsize=(5, 4))
        #     igor.plot(
        #         ax=ax,
        #         xs=[df["B_ref"].values],
        #         ys=[hallangle.values],
        #         xlabel="$B$ (T)",
        #         ylabel="Hall Angle",
        #         labels=[""],
        #         min_x=-14,
        #         max_x=14,
        #         x_step=4,
        #         sub_x_step=2,
        #         min_y=0,
        #         max_y=1,
        #         y_step=4,
        #         sub_y_step=2,
        #         suffix=f"_{sample}",
        #         savepath=savepath_hallangle,
        #     )

        #     # B=0に近いの時のホール角をdataに追加
        #     abs_ = abs(df["B_ref"] - 0)
        #     index = abs_.idxmin()
        #     hallangle_list.append(hallangle[index])
        #     temps_list.append(df["temp"][index])
        #     continue
        # else:
        df = pd.read_csv(dirpath / "temp_dep.csv")
        # plot(df["temps"], df["HallAngle0"], label="", xlabel="$T$ (K)", ylabel="Hall Angle ", title="", MIN_X=MIN_X, MAX_X=MAX_X, X_STEP=X_STEP, SUB_X_STEP=SUB_X_STEP, MIN_Y=MIN_Y, MAX_Y=MAX_Y, Y_STEP=Y_STEP, SUB_Y_STEP=SUB_Y_STEP, savepath=savepath_hallangle, hlines=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            igor.plot(
                ax=ax,
                xs=[df["temps"].values],
                ys=[df["HallAngle0"].values],
                xlabel="$T$ (K)",
                ylabel="Hall Angle",
                labels=[""],
                min_x=0,
                max_x=300,
                x_step=4,
                sub_x_step=2,
                min_y=0,
                max_y=1,
                y_step=4,
                sub_y_step=2,
                suffix=f"_{sample}",
                savepath=savepath_hallangle,
            )

            # 最小温度2Kに近いの時のホール角をdataに追加
            # abs_ = abs(df["temps"] - 2)
            # index = abs_.idxmin()
            # 最大のホール角をdataに追加
            index = df["HallAngle0"].idxmax()
            hallangle_list.append(df["HallAngle0"][index])
            temps_list.append(df["temps"][index])
        except KeyError:
            # BST（非磁性体の場合）
            hallangle_list.append(0.0)
            temps_list.append(0.0)
    savepath = savedir / "HallAngle_vs_x.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    print("hallangle_list")
    pprint(hallangle_list)
    igor.plot(
        ax=ax,
        xs=[np.array(xlist)],
        ys=[np.array(hallangle_list)],
        xlabel="x",
        ylabel="Hall Angle",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=1,
        y_step=6,
        sub_y_step=2,
        savepath=savepath,
        fmt="o-",
    )
    # savepath = savedir / "HallAngle_vs_T.png"
    return None


def plot_fwhm_vs_x(samples, xlist, dirnames, xrd_dir, savedir):
    xlist = np.array(xlist).copy()
    center_006_list, fwhm_006_list, center_0015_list, fwhm_0015_list = [], [], [], []
    for idx, (sample, dirname) in enumerate(zip(samples, dirnames)):
        if sample == "#1-1370":
            xlist = np.delete(xlist, idx)
            continue
        with open(xrd_dir / sample / "rock006_fit.txt", "r") as f:
            # 例
            # center: 8.844
            # FWHM: 0.669
            txt = f.readlines()
            center_006_list.append(float(txt[0].split(":")[1]))
            fwhm_006_list.append(float(txt[1].split(":")[1]))
        with open(xrd_dir / sample / "rock15_fit.txt", "r") as f:
            txt = f.readlines()
            center_0015_list.append(float(txt[0].split(":")[1]))
            fwhm_0015_list.append(float(txt[1].split(":")[1]))
    center_006_list = np.array(center_006_list).copy()
    center_0015_list = np.array(center_0015_list).copy()
    fwhm_006_list = np.array(fwhm_006_list).copy()
    fwhm_0015_list = np.array(fwhm_0015_list).copy()

    igor = Igor()
    fig, ax = plt.subplots(figsize=(5, 4))
    savepath = savedir / "peak_center_vs_x.png"
    print("center_006_list")
    pprint(center_006_list)
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[center_006_list],
        xlabel="x",
        ylabel="center of (006) peak",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        y_step=6,
        sub_y_step=2,
        suffix="_(006)",
        savepath=savepath,
    )
    print("center_0015_list")
    pprint(center_0015_list)
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[center_0015_list],
        xlabel="x",
        ylabel="center of (0015) peak",
        labels=[""],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        y_step=6,
        sub_y_step=2,
        suffix="_(0015)",
        savepath=savepath,
        fmt="o-",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    savepath = savedir / "peak_fwhm_vs_x.png"
    igor.plot(
        ax=ax,
        xs=[xlist, xlist],
        ys=[fwhm_006_list, fwhm_0015_list],
        xlabel="x",
        ylabel="FWHM of (006) & (0015) rocking curve",
        labels=["(006)", "(0015)"],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=1,
        y_step=6,
        sub_y_step=2,
        savepath=savepath,
        fmt="o-",
    )
    return center_006_list, center_0015_list


def plot_c_axis_length_vs_x(samples, xlist, theta_0_list, dirnames, xrd_dir, savedir):
    xlist = np.array(xlist).copy()
    c_axis_length_list = []
    for idx, (sample, dirname) in enumerate(zip(samples, dirnames)):
        c_axis_length_list.append(estimate_c_axis_length(sample, theta_0_list[idx], target="006"))
    igor = Igor()
    fig, ax = plt.subplots(figsize=(5, 4))
    savepath = savedir / "c_axis_length_vs_x.png"
    c_axis_length_list = np.array(c_axis_length_list).copy()
    print("c_axis_length_list")
    pprint(c_axis_length_list)
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[c_axis_length_list],
        xlabel="x",
        ylabel="c axis length (nm)",
        labels=["(006)"],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=3.0,
        max_y=3.1,
        y_step=6,
        sub_y_step=2,
        savepath=savepath,
        fmt="o-",
    )


def plot_growth_rate(
    samples,
    xlist,
    theta_0_list,
    xrd_dir,
    savepath,
    epilog_path="/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx",
):
    """_summary_

    Args:
        samples (_type_): _description_
        xlist (_type_): _description_
        theta_0_list (_type_): あるピークの中心の角度．006の精度が良いのでそれを使う．
        savepath (_type_): _description_
        epilog_path (str, optional): _description_. Defaults to "/Users/uedataiga/Desktop/grad-research/data_analysis/epilog/epilog_CBST.xlsx".
    """
    rocking = "006"
    xlist = np.array(xlist).copy()
    df_epilog = pd.read_excel(epilog_path, sheet_name="log", header=0, index_col=None)

    thickness_list, growth_rate_list = [], []
    for sample, theta_0 in zip(samples, theta_0_list):
        path_2tw = xrd_dir / sample / "2tw.csv"
        thickness_1order, thickness_2order = estimate_thickness_by_laue(theta_0, path_2tw, rocking=rocking)
        main_growth_time = df_epilog["main_growth_time (min)"][df_epilog["sample"] == sample].values[0]
        growth_rate = thickness_1order / main_growth_time
        print(f"thickness_1order: {thickness_1order} nm")
        print(f"thickness_2order: {thickness_2order} nm")
        print(f"growth_rate: {growth_rate} nm/min")
        # df["thickness (nm)"][df["sample"] == sample] = thickness_1order
        # df["growth_rate (nm/min)"][df["sample"] == sample] = growth_rate

        thickness_list.append(thickness_1order)
        growth_rate_list.append(growth_rate)
    thickness_list = np.array(thickness_list).copy()
    growth_rate_list = np.array(growth_rate_list).copy()
    igor = Igor(digits=2)
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(211)
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[growth_rate_list],
        xlabel="x",
        ylabel="growth rate (nm/min)",
        labels=[rocking],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=0,
        max_y=0.35,
        y_step=6,
        sub_y_step=2,
        fmt="o-",
    )
    ax = fig.add_subplot(212)
    igor.plot(
        ax=ax,
        xs=[xlist],
        ys=[thickness_list],
        xlabel="x",
        ylabel="thickness (nm)",
        labels=[rocking],
        min_x=0,
        max_x=0.1,
        x_step=6,
        sub_x_step=2,
        min_y=None,
        max_y=None,
        y_step=5,
        sub_y_step=2,
        savepath=savepath,
        fmt="o-",
    )


def estimate_TC(sample, dirname, directory, use="Ryx", T_C_threshold=0.1):
    """Rxxなら温度依存性のピーク，Ryxなら温度依存性が0を横切る点でキュリー温度を推定する

    Args:
        sample (_type_): _description_
        dirname (_type_): _description_
        use (str): "Rxx" or "Ryx"
    """
    igor = Igor()
    if use == "Rxx":
        # 正直難しいので手動でやる
        pass
    elif use == "Ryx":
        # Hall測定したときの
        min_x, max_x, x_step, sub_x_step = 0, 300, 4, 2

        dirpath = directory / f"{dirname}/{sample[3:]}"
        savepath = dirpath / f"temp_dep_{sample}.png"
        df = pd.read_csv(dirpath / "temp_dep.csv")
        if ["RyxA"] not in df.columns:
            igor.plot(
                xs=[df["temps"].values],
                ys=[[0] * len(df["temps"])],
                label=[""],
                xlabel=r"$T$ (K)",
                ylabel=r"$R_{yx}$ (Ω)",
                title=sample,
                min_x=min_x,
                max_x=max_x,
                x_step=x_step,
                sub_x_step=sub_x_step,
                min_y=-0.25,
                max_y=1.0,
                y_step=4,
                sub_y_step=2,
                savepath=savepath,
            )
            T_C = 0.0
        else:
            igor.plot(
                xs=[df["temps"].values],
                ys=[df["RyxA"].values],
                label=[""],
                xlabel=r"$T$ (K)",
                ylabel=r"$R_{yx}$ (Ω)",
                title=sample,
                min_x=min_x,
                max_x=max_x,
                x_step=x_step,
                sub_x_step=sub_x_step,
                savepath=savepath,
            )
            # # 初めrてRyxAが0を下回る温度を求める
            # # T_C = df[df["RyxA_norm"] <= T_c_threshold].iloc[0, 0]
            try:
                T_C = df[df["RyxA_norm"] <= T_C_threshold].iloc[0, 0]
            except IndexError:
                # なかった場合は温度の最小値とその温度を出力
                T_C = df["temps"].min()
                print(f"no T below {T_C_threshold} in {sample}")
        print("T_C = ", T_C)
        with open(dirpath / "TC.txt", "w") as f:
            f.write(f"T_C: {T_C}")
        return T_C


def estimate_c_axis_length(sample, theta_0, target="006"):
    """(006) or (0015)中心のthetaをθ_0，c軸長をd，x線の波長をλとすると，以下の式が成り立つ．
    d = λ / (2 * sin(θ_0 / 2))
    この式を用いてc軸長を計算する．
    参考：The unit size of a QL is 10.17  Å in the case of Bi2Te3.

    Args:
        data (Dict): _description_
    """
    xrd_lambda = 1.540598  # Kα（カッパアルファ1）の波長
    # NOTE: すごい怪しい
    d = (3 / 5) * (xrd_lambda / (2 * np.sin(np.deg2rad(theta_0))))
    if target == "006":
        d = d
    elif target == "0015":
        # NOTE: すごい怪しい
        d = d * (5 / 2)

    return d


def estimate_thickness_by_laue(theta_0, path_2tw, rocking="006"):
    """ラウエフリンジのピークから膜圧を推定する．

    Args:
        samples (_type_): _description_
        centers (_type_): _description_
        dirnames (_type_): _description_
        paths_2tw (_type_): _description_
        rocking (str, optional): _description_. Defaults to "006".
        epilog_path (str, optional): _description_. Defaults to "".
    """
    xrd_lambda = 1.540598  # Kα（カッパアルファ1）の波長
    df_2tw = pd.read_csv(path_2tw)
    if rocking == "006":
        x_range = [14, 21]
    elif rocking == "0015":
        x_range = [41, 48]
    peak_1st, _ = detect_peak(df_2tw["2theta"].values, df_2tw["int"].values, x_range=x_range, order="1")

    peak_m1st, _ = detect_peak(df_2tw["2theta"].values, df_2tw["int"].values, x_range=x_range, order="-1")
    peak_2nd, _ = detect_peak(
        df_2tw["2theta"].values,
        df_2tw["int"].values,
        x_range=x_range,
        rocking=rocking,
        order="2",
    )
    peak_m2nd, _ = detect_peak(df_2tw["2theta"].values, df_2tw["int"].values, x_range=x_range, order="-2")
    print(f"peak_1st: {peak_1st}")
    print(f"peak_m1st: {peak_m1st}")
    print(f"peak_2nd: {peak_2nd}")
    print(f"peak_m2nd: {peak_m2nd}")

    thickness_01 = 3 * xrd_lambda / (4 * abs((np.sin(np.deg2rad(peak_1st)) - np.sin(np.deg2rad(theta_0)))))
    thickness_0m1 = 3 * xrd_lambda / (4 * abs((np.sin(np.deg2rad(peak_m1st)) - np.sin(np.deg2rad(theta_0)))))
    thickness_12 = xrd_lambda / (2 * (abs(np.sin(np.deg2rad(peak_2nd)) - np.sin(np.deg2rad(theta_0)))))
    thickness_m1m2 = xrd_lambda / (2 * (abs(np.sin(np.deg2rad(peak_m2nd)) - np.sin(np.deg2rad(theta_0)))))
    # Crが多いと誤差（特に2次）が大きくなる．
    thickness_1order = (thickness_01 + thickness_0m1) / 2
    thickness_2order = (thickness_12 + thickness_m1m2) / 2

    return thickness_1order, thickness_2order


if __name__ == "__main__":
    main()
