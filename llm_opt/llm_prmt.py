import json
import os
from pathlib import Path
from typing import Any, Dict

from . import llm_config as config

from .llm_loader import (
    load_factor_select,
    load_llm_factor_template,
    load_operators_doc,
    load_backtest_stats_text,
    load_backtest_metrics_doc,
)

PREFIX_PROMPT = (
    "You are a quantitative finance researcher specializing in alpha factor "
    "development and optimization. Analyze and improve the supplied factors.\n"
    "1. Read the candidate factor pool, explain each factor's mathematical logic "
    "and financial interpretation, and fill in info.name and info.desc.\n"
    "2. Review the training and test backtest results, focusing on ExcessIC, "
    "annualized_long_excess, and sharpe_long_excess. Use the metric reference "
    "to identify weaknesses and possible improvements.\n"
    "3. Preserve each original factor's core logic and propose nine distinct "
    "variations of its mathematical components. Prioritize performance on both "
    "in-sample and out-of-sample data and the ability to generalize.\n"
    "4. Build expressions using only the documented operators. Follow the JSON "
    "template and output requirements, and include every original factor.\n"
)

GUIDELINES = (
    "[Field guidelines]\n"
    "- enabled: Boolean; default true.\n"
    "- idx: String factor index. Append .1 through .9 for the nine variants; "
    "for example, 10.1 is the first variant of factor 10.\n"
    "- expr: Factor expression using the same syntax as the input.\n"
    "- info.name: English name reflecting the mathematical logic, financial "
    "interpretation, and optimization direction.\n"
    "- info.desc: English description explaining the mathematical logic, "
    "financial interpretation, and the key changes in the variant.\n"
)

JSON_RULES = (
    "[Output requirements]\n"
    "- Return only a JSON array of objects. Do not include explanations, "
    "non-JSON text, or Markdown code fences.\n"
    "- Use strict UTF-8 JSON: double-quoted keys and strings, no raw control "
    "characters, comments, or trailing commas.\n"
    "- Each object must contain enabled (boolean, default true), idx (string), "
    "expr (string), and info (object with name and desc strings).\n"
    "- For each original factor N, return ten entries: N and N.1 through N.9.\n"
    "- Use only operator names in the operator reference. Do not invent "
    "functions or symbols, and preserve the input variable names.\n"
    "- Match the documented argument counts, such as ts_macd(x,N) and "
    "ts_corr(x,y,N).\n"
    "- Close all parentheses. CSRank(Abs($close-$open)) is valid; "
    "CSRank(Abs($close-$open) is invalid.\n"
    "- Write info.name and info.desc entirely in English, using precise "
    "financial terminology consistent with the expression.\n"
    "- If no reasonable variant is available, set enabled=false and briefly "
    "explain the reason in info.desc.\n"
    "- Format the array with two-space indentation and each field on its "
    "own line.\n"
)


def build_prompt(run_dir: str) -> str:
    data = load_factor_select(run_dir)
    data_str = json.dumps(data, ensure_ascii=False, indent=2)

    tmpl = load_llm_factor_template()
    tmpl_str = json.dumps(tmpl, ensure_ascii=False, indent=2)

    ops_md = load_operators_doc()

    backtest_stats = load_backtest_stats_text(run_dir)
    backtest_metrics = load_backtest_metrics_doc()

    return f"""

{PREFIX_PROMPT}

==============================
    Factor pool to be optimized
==============================

{data_str}

==============================
    Backtest data performance
==============================

{backtest_stats}

==============================
    Backtest indicator description
==============================

{backtest_metrics}

==============================
    Standard format template
==============================

{tmpl_str}

{GUIDELINES}

==============================
    Operator documentation
==============================

{ops_md}

==============================
    Output format requirements
==============================

{JSON_RULES}

"""


def main() -> None:
    prompt = build_prompt(config.RUN_DIR)
    print(prompt)


if __name__ == "__main__":
    main()
