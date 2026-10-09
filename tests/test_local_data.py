"""Offline regression tests for the public local-data workflow."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from paper_trading.local_data import LocalCSVData
from paper_trading.pt_converter import PTConverter
from paper_trading.pt_fetcher import PTFetcher
from paper_trading.pt_main import export_pool


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provider = LocalCSVData(self.root)

    def write_data(self, name="DEMO", columns=None):
        frame = pd.DataFrame({
            "time": ["2024-01-03", "2024-01-02", "2024-01-03"],
            "open": [101, 100, 102], "high": [103, 102, 104],
            "low": [100, 99, 101], "close": [102, 101, 103],
            "volume": [1200, 1000, 1300],
        })
        if columns is not None:
            frame = frame[columns]
        frame.to_csv(self.root / f"{name}.csv", index=False)

    def test_local_csv_is_sorted_and_deduplicated(self):
        self.write_data()
        result = self.provider.get_daily_data("DEMO")
        self.assertEqual(result["close"].tolist(), [101, 103])
        self.assertTrue(result["time"].is_monotonic_increasing)

    def test_missing_required_columns_are_rejected(self):
        self.write_data(columns=["time", "close"])
        with self.assertRaisesRegex(ValueError, "Missing daily-data columns"):
            self.provider.get_daily_data("DEMO")

    def test_invalid_instrument_and_external_symlink_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "file-safe"):
            self.provider.get_daily_data("../private")
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "external.csv"
            target.write_text("time,close\n2024-01-01,100\n")
            (self.root / "DEMO.csv").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "inside the data directory"):
                self.provider.get_daily_data("DEMO")

    def test_fetch_and_convert_preserve_local_prices(self):
        self.write_data()
        config = SimpleNamespace(data=SimpleNamespace(
            max_stocks=None, max_days=2, max_backtrack_days=0, max_future_days=0,
        ))
        frame = PTFetcher(self.provider).get_all_day01_data(
            pd.DataFrame({"tr_code": ["DEMO"]}), config)
        stock_data = PTConverter().convert_day01_to_qlib_format(frame, config)
        self.assertEqual(stock_data.stock_codes, ["DEMO"])
        self.assertEqual(stock_data.close[:, 0].tolist(), [101, 103])
        self.assertEqual(stock_data.data.shape, (2, 5, 1))

    def test_missing_instrument_file_is_not_silently_skipped(self):
        config = SimpleNamespace(data=SimpleNamespace(max_stocks=None, max_days=2))
        with self.assertRaises(FileNotFoundError):
            PTFetcher(self.provider).get_all_day01_data(pd.DataFrame({"tr_code": ["MISSING"]}), config)

    def test_stock_pool_export_is_local_and_rejects_path_names(self):
        frame = pd.DataFrame({"tr_code": ["DEMO"], "mark": [100.0]})
        path = export_pool("research", frame, self.root / "exports")
        pd.testing.assert_frame_equal(pd.read_csv(path), frame)
        with self.assertRaisesRegex(ValueError, "plain file names"):
            export_pool("../private", frame, self.root)


if __name__ == "__main__":
    unittest.main()
