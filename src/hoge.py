def analyze_Hall_multi_temps(self, dfs):
    """ """
    analyzed_dfs = {}  # 各温度における解析済みデー
    raw_dfs = {}  # 各温度における生データ
    summaries = pd.DataFrame()  # 各温度点の特徴量をまとめたデータ(indexは温度点)

    ######################
    # 昔の処理
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
    ######################
