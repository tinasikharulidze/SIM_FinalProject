import numpy as np
import pandas as pd


class GalaxyZooPreprocessor:
    """
    Complete Galaxy Zoo preprocessing pipeline:
    - Remove unwanted columns
    - Extinction correction
    - Color indices
    - Concentration index
    - Surface brightness
    - Log-transforms of radii and flux-like variables
    - Error encoding correction
    - Drop NaNs
    - Remove extreme outliers
    """

    def __init__(self):
        self.unwanted_columns = [
            "PETROR50_R_KPC_SIMPLE_BIN",
            "PETROMAG_MR_SIMPLE_BIN",
            "REDSHIFT_SIMPLE_BIN",
            "WVT_BIN",
            "ROWC_U",
            "COLC_U",
            "ROWC_G",
            "COLC_G",
            "ROWC_R",
            "COLC_R",
            "ROWC_I",
            "COLC_I",
            "ROWC_Z",
            "COLC_Z",
            "RUN",
            "RERUN",
            "CAMCOL",
            "FIELD",
            "OBJ",
            "RA",
            "DEC",
            "REGION",
        ]

        self.flux_like_cols = [
            "PETROR50_R",
            "PETROR90_R",
            "PETROR50_R_KPC",
            "PETROMAGERR_U",
            "PETROMAGERR_G",
            "PETROMAGERR_R",
            "PETROMAGERR_I",
            "PETROMAGERR_Z",
            "PETROMAGERR_MU",
            "PETROMAGERR_MG",
            "PETROMAGERR_MR",
            "PETROMAGERR_MI",
            "PETROMAGERR_MZ",
            "DEVMAGERR_R",
            "EXPMAGERR_R",
            "CMODELMAGERR_R",
        ]

        self.error_columns = [
            "PETROMAGERR_U",
            "PETROMAGERR_G",
            "PETROMAGERR_R",
            "PETROMAGERR_I",
            "PETROMAGERR_Z",
            "PETROMAGERR_MU",
            "PETROMAGERR_MG",
            "PETROMAGERR_MR",
            "PETROMAGERR_MI",
            "PETROMAGERR_MZ",
            "DEVMAGERR_R",
            "EXPMAGERR_R",
            "CMODELMAGERR_R",
        ]

        self.extinction_cols = [
            ("PETROMAG_U", "EXTINCTION_U"),
            ("PETROMAG_G", "EXTINCTION_G"),
            ("PETROMAG_R", "EXTINCTION_R"),
            ("PETROMAG_I", "EXTINCTION_I"),
            ("PETROMAG_Z", "EXTINCTION_Z"),
        ]

    # ---------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------
    def preprocess(self, df):
        df = df.copy()

        df = self.remove_unwanted_columns(df)
        df = self.correct_for_extinction(df)
        df = self.correct_petromagerror_encoding(df)

        df = self.add_color_indices(df)
        df = self.add_concentration_index(df)
        df = self.add_surface_brightness(df)
        df = self.add_log_radii(df)
        df = self.add_flux_logs(df)

        df = self.drop_nan_rows(df)
        df = self.remove_extreme_outliers(df)

        df = self.drop_id(df)

        return df

    # ---------------------------------------------------
    # PIPELINE STEPS
    # ---------------------------------------------------

    def remove_unwanted_columns(self, df):
        df = df.drop(
            columns=[c for c in self.unwanted_columns if c in df.columns],
            errors="ignore",
        )
        return df

    def correct_for_extinction(self, df):
        df = df.copy()
        corrected_cols = []

        for mag_col, ext_col in self.extinction_cols:
            if mag_col in df.columns and ext_col in df.columns:
                corrected_col = mag_col + "_corr"
                df[corrected_col] = df[mag_col] - df[ext_col]
                corrected_cols.append(corrected_col)

        # Drop original magnitudes and extinction columns
        cols_to_drop = [mag for mag, _ in self.extinction_cols] + [
            ext for _, ext in self.extinction_cols
        ]
        df = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
        )

        return df

    def correct_petromagerror_encoding(self, df):
        for col in self.error_columns:
            if col in df.columns:
                df[col] = df[col].replace(99999.0, np.nan)
        return df

    def add_color_indices(self, df):
        pairs = [
            ("PETROMAG_U", "PETROMAG_G"),
            ("PETROMAG_G", "PETROMAG_R"),
            ("PETROMAG_R", "PETROMAG_I"),
            ("PETROMAG_I", "PETROMAG_Z"),
        ]
        for c1, c2 in pairs:
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_{c2}_color"] = df[c1] - df[c2]
        return df

    def add_concentration_index(self, df):
        if "PETROR90_R" in df.columns and "PETROR50_R" in df.columns:
            df["CONC_R"] = df["PETROR90_R"] / df["PETROR50_R"].replace(0, np.nan)
        return df

    def add_surface_brightness(self, df):
        if "PETROMAG_R" in df.columns and "PETROR50_R" in df.columns:
            r50 = df["PETROR50_R"].replace(0, np.nan)
            df["SURFACE_BRIGHTNESS_R"] = df["PETROMAG_R"] + 2.5 * np.log10(
                2 * np.pi * (r50**2)
            )
        return df

    def add_log_radii(self, df):
        df = df.copy()

        if "PETROR50_R" in df.columns:
            df["LOG_PETROR50_R"] = np.log1p(df["PETROR50_R"])
        if "PETROR90_R" in df.columns:
            df["LOG_PETROR90_R"] = np.log1p(df["PETROR90_R"])
        if "PETROR50_R_KPC" in df.columns:
            df["LOG_PETROR50_R_KPC"] = np.log1p(df["PETROR50_R_KPC"])

        # Drop original radius columns
        cols_to_drop = ["PETROR50_R", "PETROR90_R", "PETROR50_R_KPC"]
        df = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
        )

        return df

    def add_flux_logs(self, df):
        df = df.copy()

        for col in self.flux_like_cols:
            if col in df.columns:
                df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

        # Drop original flux-like columns
        df = df.drop(
            columns=[c for c in self.flux_like_cols if c in df.columns], errors="ignore"
        )

        return df

    def drop_nan_rows(self, df):
        return df.dropna()

    def remove_extreme_outliers(self, df, features=None, threshold=10.0):
        df_clean = df.copy()
        if features is None:
            features = df.select_dtypes(include="number").columns.tolist()
        if "OBJID" in features:
            features.remove("OBJID")

        outlier_mask = pd.Series(False, index=df.index)
        for col in features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outlier_mask |= (df[col] < lower) | (df[col] > upper)

        return df_clean[~outlier_mask]

    def drop_id(self, df):
        if "OBJID" in df.columns:
            df = df.drop(columns=["OBJID"])
        return df
