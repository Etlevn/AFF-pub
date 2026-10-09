"""Load user-supplied daily market data from local CSV files."""

from pathlib import Path
import re

import pandas as pd


class LocalCSVData:
    """Read <instrument>.csv with time, open, high, low, close, and volume."""

    COLUMNS = ("time", "open", "high", "low", "close", "volume")

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()

    def get_daily_data(self, instrument: str) -> pd.DataFrame:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", instrument):
            raise ValueError("Instrument names must be plain file-safe identifiers")
        path = (self.directory / f"{instrument}.csv").resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Market data must be located inside the data directory")
        frame = pd.read_csv(path)
        missing = set(self.COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing daily-data columns: {', '.join(sorted(missing))}")
        frame = frame.loc[:, list(self.COLUMNS)].copy()
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        if frame["time"].isna().any():
            raise ValueError("Daily data contains missing dates")
        for column in self.COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
