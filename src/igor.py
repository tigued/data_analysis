import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Union, Tuple, Dict, Any, Optional

# 目盛りの上限，下限用
# from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, getcontext
import decimal
import math

# getcontext().prec = 3
# ROUND = 2
# def round_down(val, digits=0):
#     val = float(val)  # conversion from numpy.int64 to Decimal is not supported
#     if val >= 0:
#         return float(
#             Decimal(val).quantize(Decimal("1e%s" % (-digits)), rounding=ROUND_DOWN)
#         )
#     else:
#         return float(
#             Decimal(val).quantize(Decimal("1e%s" % (-digits)), rounding=ROUND_UP)
#         )


# def round_up(val, digits=0):
#     val = float(val)  # conversion from numpy.int64 to Decimal is not supported
#     if val >= 0:
#         return float(
#             Decimal(val).quantize(Decimal("1e%s" % (-digits)), rounding=ROUND_UP)
#         )
#     else:
#         return float(
#             Decimal(val).quantize(Decimal("1e%s" % (-digits)), rounding=ROUND_DOWN)
#         )


# def search_min_max_step_substep(vals: np.array, mn = None, mx = None, substep_num: int = 2):
#     """良さげな最小値，最大値，ステップ，サブステップを探す
#     未検証

#     Args:
#         vals (np.array): _description_
#         substep_num (int, optional): _description_. Defaults to 2.

#     Raises:
#         ValueError: _description_

#     Returns:
#         _type_: _description_
#     """
#     if mn is None and mx is None:
#         rnd = 1
#         val_min = np.min(vals)
#         val_max = np.max(vals)
#         val_down = round_down(val_min, digits=rnd)
#         val_up = round_up(val_max, digits=rnd)
#         while (rnd < 10) or ((val_max - val_min) / (val_up - val_down) < z2):
#             val_down = round_down(val_min, digits=rnd)
#             val_up = round_up(val_max, digits=rnd)
#             rnd += 1
#         if rnd == 10:
#             raise ValueError("Error: too many digits")
#     elif mn is None and mx is not None: # 工事中
#         pass
#     elif mn is not None and mx is None: # 工事中
#         pass
#     else: # 工事中
#         pass
#     # min_str, max_str = str(val_down), str(val_up)
#     # idx = 0
#     # while min_str[idx] != max_str[idx]:
#     #     idx += 1
#     step_candidate = [4, 3, 2]
#     step = None
#     for stp in step_candidate:
#         if float(val_up - val_down)*(rnd+1) % stp == 0:
#             step = stp
#             break
#     if step is None:
#         step = 5
#     substep = step / substep_num
#     return val_down, val_up, step, substep


# TODO: matplotlibの継承で書き直す
class Igor:
    """Igorそっくりのグラフを作成する"""

    def __init__(
        self,
        cmap="jet",
        dpi=300,
        digits=4,
        font="Arial",
        fontsize=22,
        major_tick_len=10,
        minor_tick="xy",  # "x", "y", "xy
        minor_tick_len=5,
        linewidth=1,
        margin=0.05,
        *args,
        **kwargs,
    ):
        self.cmap = plt.get_cmap(cmap)
        self.dpi = dpi
        self.digits = digits
        self.fontsize = fontsize
        self.margin = margin
        decimal.getcontext().prec = digits
        plt.rcParams["font.family"] = font  # "Times New Roman"      #全体のフォントを設定
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ
        if "x" in minor_tick:
            plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        if "y" in minor_tick:
            plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        # plt.rcParams["xtick.major.width"] = 1.5  # x軸主目盛り線の線幅
        # plt.rcParams["ytick.major.width"] = 1.5  # y軸主目盛り線の線幅
        # plt.rcParams["xtick.minor.width"] = 1.0  # x軸補助目盛り線の線幅
        # plt.rcParams["ytick.minor.width"] = 1.0  # y軸補助目盛り線の線幅
        plt.rcParams["xtick.major.size"] = major_tick_len  # x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = major_tick_len  # y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = minor_tick_len  # x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = minor_tick_len  # y軸補助目盛り線の長さ
        plt.rcParams["font.size"] = fontsize  # フォントの大きさ
        plt.rcParams["axes.linewidth"] = int(linewidth * self.fontsize / 18)  # 囲みの太さ
        # plt.rcParams["figure.figsize"] = [5, 4]
        # 目盛方向を両側, 目盛の長さを5ポイント, 目盛と目盛ラベルの色
        # ax.tick_params(direction="in", length=10, colors="black", width=1)
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True

    def plot(
        self,
        ax,
        xs: np.array,
        ys: np.array,
        xlabel: str,
        ylabel: str,
        labels: List[str] = [""],
        label_pos="best",
        # fmt="-",
        min_x=None,
        max_x=None,
        x_step=6,
        sub_x_step=2,
        min_y=None,
        max_y=None,
        scale_x=1,
        scale_y=1,
        y_step=6,
        sub_y_step=2,
        log_scale="",  # "x", "y", "xy"
        grid="",  # "x", "y", "xy"
        omit_tick="",  # "x", "y", "xy"
        title="",
        suffix="",
        savepath="",
        fmt="-",
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            ax (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            xlabel (_type_): _description_
            ylabel (_type_): _description_
            label (str, optional): _description_. Defaults to "".
            label_pos (str, optional): _description_. 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
            min_x (_type_, optional): _description_. Defaults to None.
            max_x (_type_, optional): _description_. Defaults to None.
            x_step (int, optional): _description_. Defaults to 5.
            sub_x_step (int, optional): _description_. Defaults to 2.
            min_y (_type_, optional): _description_. Defaults to None.
            max_y (_type_, optional): _description_. Defaults to None.
            y_step (int, optional): _description_. Defaults to 5.
            sub_y_step (int, optional): _description_. Defaults to 2.
            grid (str, optional): _description_. Defaults to "".
            title (str, optional): _description_. Defaults to "".
            suffix (str, optional): _description_. Defaults to "".
            savepath (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        assert (
            # type(ax) is mpl.axes._subplots.AxesSubplot
            issubclass(type(ax), mpl.axes.SubplotBase)
        ), f"ax must be AxesSubplot: {type(ax)}"
        assert len(xs) == len(ys) == len(labels), f"len(xs), len(ys), len(labels): {len(xs)}, {len(ys)}, {len(labels)}"
        for idx, (x, y) in enumerate(zip(xs, ys)):
            assert len(x) == len(y), f"{idx}番目のデータ：len(x), len(y): {len(x)}, {len(y)}"
        # if type(xs) is list:
        #     xs = np.array(xs)  # NOTE: 複数のデータの点数が異なるときにエラーが出る
        # if type(ys) is list:
        #     ys = np.array(ys)
        # assert (
        #     len(xs.shape) == 2 and xs.shape == ys.shape
        # ), f"xs.shape, ys.shape: {xs.shape}, {ys.shape}"
        xs_new, ys_new = [], []
        for idx, (x, y) in enumerate(zip(xs, ys)):
            # xかyにnanがある場合は削除
            try:
                remain_idx = ~np.isnan(np.array(x).astype(float)) & ~np.isnan(np.array(y).astype(float))
            except TypeError as e:
                print(e)
            xs_new.append(x[remain_idx] * scale_x)
            ys_new.append(y[remain_idx] * scale_y)

        cmap_idx = np.linspace(0, 1, len(xs))
        for idx, (x, y, label) in enumerate(zip(xs_new, ys_new, labels)):
            try:
                ax.plot(
                    x,
                    y,
                    fmt,
                    linewidth=1.5,
                    label=label,
                    color=self.cmap(cmap_idx[idx]),
                )
            except ValueError:
                print(f"{idx}番目のデータが空")
                continue

        tmp_min_x, tmp_max_x = 1e16, -1e16
        tmp_min_y, tmp_max_y = 1e16, -1e16
        for x in xs_new:
            try:
                tmp_min_x = min(tmp_min_x, np.min(x))
                tmp_max_x = max(tmp_max_x, np.max(x))
            except ValueError:
                continue
        for y in ys_new:
            # print("min, max", np.min(y), np.max(y))
            try:
                tmp_min_y = min(tmp_min_y, np.min(y))
                tmp_max_y = max(tmp_max_y, np.max(y))
            except ValueError:
                continue
        diff_x = tmp_max_x - tmp_min_x
        diff_y = tmp_max_y - tmp_min_y
        if min_x is None:
            # min_x = round_down(np.min(xs), digits=self.digits)
            min_x = tmp_min_x - self.margin * diff_x
            # min_x = +decimal.Decimal(np.sign(x_min) * 1.1 * abs(x_min))
            try:
                min_x = round(min_x, self.digits - math.floor(math.log10(abs(min_x))) - 1)
            except (ValueError, OverflowError):
                min_x = 1e-16
        if max_x is None:
            # max_x = round_up(np.max(xs), digits=self.digits)
            max_x = tmp_max_x + self.margin * diff_x
            # max_x = +decimal.Decimal(np.sign(max_x) * 1.1 * abs(max_x))
            try:
                max_x = round(max_x, self.digits - math.floor(math.log10(abs(max_x))) - 1)
            except (ValueError, OverflowError):
                max_x = 1e-16
        if min_y is None:
            # min_y = round_down(np.min(ys), digits=self.digits)
            min_y = tmp_min_y - self.margin * diff_y
            # min_y = +decimal.Decimal(np.sign(y_min) * 1.1 * abs(y_min))
            try:
                min_y = round(min_y, self.digits - math.floor(math.log10(abs(min_y))) - 1)
            except (ValueError, OverflowError):
                min_y = 1e-16
        if max_y is None:
            # max_y = round_up(np.max(ys), digits=self.digits)
            max_y = tmp_max_y + self.margin * diff_y
            # max_y = +decimal.Decimal(np.sign(max_y) * 1.1 * abs(max_y))
            try:
                max_y = round(max_y, self.digits - math.floor(math.log10(abs(max_y))) - 1)
            except (ValueError, OverflowError):
                max_y = 1e-16
        # assert min_x != max_x, f"min_x == max_x: {min_x} == {max_x}．有効数字を増やしてください"
        # assert min_y != max_y, f"min_y == max_y: {min_y} == {max_y}．有効数字を増やしてください"
        if min_x == max_x:
            min_x -= 0.01
            max_x += 0.01
        if min_y == max_y:
            min_y -= 0.01
            max_y += 0.01

        max_x += 1e-16
        max_y += 1e-16

        if labels[0] != "":
            # label_posがstrなら
            if type(label_pos) is str:
                ax.legend(loc=label_pos, frameon=False, framealpha=0, fontsize=self.fontsize)
            # label_posがtupleなら
            elif type(label_pos) is tuple:
                ax.legend(
                    bbox_to_anchor=label_pos,
                    borderaxespad=1,
                    loc="upper left",
                    frameon=False,
                    framealpha=1,
                    fontsize=self.fontsize,
                )
        ax.set_title(title)
        if "x" in log_scale:
            ax.set_xscale("log")
            xticks = np.logspace(np.log10(min_x), np.log10(max_x), x_step)
        else:
            xticks = np.linspace(min_x, max_x, x_step)
        if "y" in log_scale:
            ax.set_yscale("log")
            yticks = np.logspace(np.log10(min_y), np.log10(max_y), y_step)
            print("yticks", yticks)
        else:
            yticks = np.linspace(min_y, max_y, y_step)

        ax.tick_params(
            which="major",
            direction="in",
            length=5 * self.fontsize / 18,
            colors="black",
            labelsize=self.fontsize,
        )
        if "x" in omit_tick:
            # ax.set_xticks([])
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(xlabel, fontsize=self.fontsize)
            ax.set_xticks(xticks, labelsize=self.fontsize)
        ax.set_xlim(min_x, max_x)
        if "y" in omit_tick:
            # ax.set_yticks([])
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel(ylabel, fontsize=self.fontsize)
            ax.set_yticks(yticks, labelsize=self.fontsize)
        ax.set_ylim(min_y, max_y)

        # 副目盛りの設定
        # sub_x_step = x_step * sub_x_step
        # sub_y_step = y_step * sub_y_step
        ax.tick_params(
            which="minor",
            direction="in",
            length=2.5 * self.fontsize / 18,
            colors="black",
            labelsize=self.fontsize,
        )
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(sub_x_step))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(sub_y_step))
        # ax.set_xticks(np.linspace(min_x, max_x, step=sub_x_step), minor=True)
        # ax.set_yticks(np.linspace(min_y, max_y, step=sub_y_step), minor=True)
        if "x" in grid:
            ax.grid(axis="x")
        if "y" in grid:
            ax.grid(axis="y")

        if savepath != "":
            savepath = str(savepath).replace(".png", f"{suffix}.png")
            print(f"savepath: {savepath}")
            try:
                plt.savefig(
                    savepath,
                    dpi=self.dpi,
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=1,
                )
            except ValueError as e:
                print(f"savepath: = {savepath} occured error when saving")

        return None

    # def plot_multiple(
    #     self,
    #     ax,
    #     xs,
    #     ys,
    #     xlabel,
    #     ylabel,
    #     labels,
    #     label_pos="left_upper",
    #     min_x=None,
    #     max_x=None,
    #     x_step=5,
    #     sub_x_step=2,
    #     min_y=None,
    #     max_y=None,
    #     y_step=5,
    #     sub_y_step=2,
    #     grid="",
    #     title="",
    #     suffix="",
    #     savepath="",
    #     *args,
    #     **kwargs,
    # ):
    #     if min_x is None:
    #         min_x = round_down(np.min(xs), digits=self.digits)
    #     if max_x is None:
    #         max_x = round_up(np.max(xs), digits=self.digits)
    #     if min_y is None:
    #         min_y = round_down(np.min(ys), digits=self.digits)
    #     if max_y is None:
    #         max_y = round_up(np.max(ys), digits=self.digits)

    #     for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
    #         ax.plot(x, y, linewidth=1.5, label=label, color=self.cmap(idx / len(xs)))

    #     max_x += 1e-16
    #     max_y += 1e-16
    #     print(
    #         "setting",
    #         min_x,
    #         max_x,
    #         min_y,
    #         max_y,
    #         x_step,
    #         y_step,
    #         sub_x_step,
    #         sub_y_step,
    #     )
    #     ax.legend(loc=label_pos)
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)
    #     ax.set_xlim(min_x, max_x)
    #     ax.set_ylim(min_y, max_y)
    #     ax.set_xticks(np.linspace(min_x, max_x, x_step))
    #     ax.set_yticks(np.linspace(min_y, max_y, y_step))
    #     # 副目盛りの設定
    #     sub_x_step = x_step * sub_x_step
    #     sub_y_step = y_step * sub_y_step
    #     ax.minorticks_on()
    #     ax.tick_params(which="minor", direction="in", length=5, colors="black")
    #     ax.set_xticks(np.linspace(min_x, max_x, step=sub_x_step), minor=True)
    #     ax.set_yticks(np.linspace(min_y, max_y, step=sub_y_step), minor=True)
    #     if grid == "xy":
    #         ax.grid()
    #     elif grid == "x":
    #         ax.grid(axis="x")
    #     elif grid == "y":
    #         ax.grid(axis="y")
    #     if savepath != "":
    #         savepath = str(savepath).replace(".png", f"{suffix}.png")
    #         plt.savefig(savepath, dpi=self.dpi, transparent=True)

    #     return None
