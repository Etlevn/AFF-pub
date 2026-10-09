"""Step 1: load daily market data from a local provider."""

import os
import pandas as pd
from tqdm import tqdm

from paper_trading.local_data import LocalCSVData
from paper_trading.pt_config import PipelineConfig


class PTFetcher:
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or LocalCSVData(
            os.environ.get("AFF_DATA_DIR", "data/daily")
        )

    def get_all_day01_data(self, df_info: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
        if df_info is None or df_info.empty:
            raise ValueError("Instrument metadata is empty")
        if not {"tr_code", "bond_code"}.intersection(df_info.columns):
            raise ValueError("Instrument metadata requires tr_code or bond_code")
        info = df_info.head(config.data.max_stocks) if config.data.max_stocks is not None else df_info
        frames = []
        for _, row in tqdm(info.iterrows(), total=len(info), desc="Loading daily data", unit="instrument"):
            if "tr_code" in info.columns:
                instrument = str(row["tr_code"])
            else:
                code = str(row["bond_code"])
                instrument = code + (".SH" if code.startswith("11") else ".SZ")
            frame = self.data_provider.get_daily_data(instrument)
            if config.data.max_days is not None:
                frame = frame.tail(config.data.max_days)
            if frame.empty:
                raise ValueError(f"No daily data for instrument: {instrument}")
            frame = frame.copy()
            frame["tr_code"] = instrument
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
