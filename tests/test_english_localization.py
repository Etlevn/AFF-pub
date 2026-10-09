"""Regression checks for English schemas and historical data compatibility."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backtest.metric_schema import normalize_metric_columns
from paper_trading.pt_mask import PTMask


ROOT = Path(__file__).resolve().parents[1]


def load_selection_module():
    path = ROOT / "factor_pipeline" / "3_factor_select.py"
    spec = importlib.util.spec_from_file_location("factor_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnglishLocalizationTests(unittest.TestCase):


    def test_legacy_metric_headers_and_split_prefixes_are_normalized(self):
        original = pd.DataFrame({
            "factor": ["1"], "\u8d85\u989dIC": [-0.3],
            "test_\u590f\u666e_long_excess": [1.2],
            "train_\u80dc\u7387_long_abs": [0.6],
        })
        normalized = normalize_metric_columns(original)
        self.assertEqual(list(normalized.columns),
                         ["factor", "ExcessIC", "test_sharpe_long_excess",
                          "train_win_rate_long_abs"])
        self.assertEqual(normalized.loc[0, "ExcessIC"], -0.3)
        pd.testing.assert_frame_equal(normalize_metric_columns(normalized), normalized)

    def test_ambiguous_metric_columns_are_rejected(self):
        frame = pd.DataFrame({"\u8d85\u989dIC": [0.1], "ExcessIC": [0.2]})
        with self.assertRaisesRegex(ValueError, "Duplicate metric columns"):
            normalize_metric_columns(frame)

    def test_selection_reads_legacy_csv_and_ranks_absolute_excess_ic(self):
        selection = load_selection_module()
        frame = pd.DataFrame({
            "factor": ["1", "2"], "\u8d85\u989dIC": [-0.4, 0.2],
            "\u5e74\u5316_long_excess": [0.3, 0.1],
            "\u590f\u666e_long_excess": [1.5, 0.7],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backtest.csv"
            frame.to_csv(path, index=False)
            loaded = selection._load_table(str(path))
        self.assertIsNotNone(loaded)
        self.assertTrue(set(selection.METRIC_WEIGHTS).issubset(loaded.columns))
        ranked = selection._rank_scores(loaded, selection.METRIC_WEIGHTS)
        self.assertLess(ranked.loc["1"], ranked.loc["2"])

    def test_bond_metadata_applies_listing_and_suspension_masks(self):
        processor = PTMask()
        processor.cb_info = pd.DataFrame({
            "bond_code": ["123456"],
            "stock_code": ["600001"],
            "subscription_date": [1704067200000],
        })
        dates = pd.to_datetime(["2023-12-31", "2024-01-01"])
        mask = pd.DataFrame(1, index=dates, columns=["123456.SZ"])
        listing = processor._apply_listing_mask(mask.copy())
        self.assertEqual(listing.iloc[:, 0].tolist(), [0, 1])
        processor.tp_info = pd.DataFrame({
            "time": [pd.Timestamp("2024-01-01")], "tp_codes": [["600001.SH"]],
        })
        suspension = processor._apply_tp_mask(mask.copy(), pd.Timestamp("2024-01-01"))
        self.assertEqual(suspension.iloc[:, 0].tolist(), [0, 0])

    def test_prompt_requires_english_and_keeps_original_and_nine_variants(self):
        from llm_opt import llm_prmt

        candidates = [{"enabled": True, "idx": "10", "expr": "Abs($close)",
                       "info": {"name": "Absolute price", "desc": "Input factor"}}]
        with patch.object(llm_prmt, "load_factor_select", return_value=candidates), \
             patch.object(llm_prmt, "load_backtest_stats_text",
                          return_value="factor,test_ExcessIC\n10,0.2"):
            prompt = llm_prmt.build_prompt("example")
        self.assertIn('"idx": "10"', prompt)
        self.assertIn("N.1 through N.9", prompt)
        self.assertIn("entirely in English", prompt)
        self.assertNotIn("must be in Chinese", prompt)
        self.assertIn("Abs($close)", prompt)
        self.assertIn("test_ExcessIC", prompt)


if __name__ == "__main__":
    unittest.main()
