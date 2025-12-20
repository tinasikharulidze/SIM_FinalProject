import pandas as pd


def load():
    # Read directly with pandas
    zoo = pd.read_csv("zoo.gz", compression="gzip")

    # Read Metadata
    r = pd.read_csv("redshift.gz")

    # Merge
    main = pd.merge(zoo, r, left_on="dr7objid", right_on="OBJID", how="inner")

    selected_columns = [
        "t01_smooth_or_features_a01_smooth_debiased"
    ] + r.columns.tolist()

    df = main[selected_columns]

    df = df.rename(columns={"t01_smooth_or_features_a01_smooth_debiased": "target"})

    return df
