import numpy as np
import pandas as pd


def spacetxt2csv(txt_file_path):
    """空白区切のテキストファイルをCSVファイルに変換する

    Args:
        txt_file_path (str): txtファイルのパス
    """
    with open(txt_file_path) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    df = pd.DataFrame(lines)
    df.to_csv(txt_file_path.replace('.txt', '.csv'), index=False, header=False)
