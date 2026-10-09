# AFF Project Overview

```text
AFF/
├── README.md                     # Setup, research workflows, and entry points
├── train_AFF.py                  # Sliding-window Alpha Factor Factory launcher
├── train_DSO.py                  # Deep symbolic optimization launcher
├── train_GP.py                   # Genetic programming launcher
├── train_RL.py                   # Reinforcement-learning launcher
├── aff_module.py                 # AFF collection, rebalancing, evaluation, and training
├── requirements/                 # Dependencies for each subsystem
│   ├── req_aff.txt               # AFF
│   ├── req_dso.txt               # DSO
│   ├── req_gp.txt                # GP
│   ├── req_rl.txt                # RL
│   └── req_dft.txt               # LLM and paper-trading integrations
├── alphagen/                     # Expression trees, factor pools, and RL
│   ├── config.py                 # Operator tags and global constants
│   ├── data/
│   │   ├── calculator.py         # Abstract IC/RankIC evaluation interfaces
│   │   ├── expression.py         # Expression nodes and evaluation
│   │   ├── tokens.py             # Feature, operator, constant, and time-offset tokens
│   │   └── tree.py               # Expression construction and validation
│   ├── models/
│   │   ├── alpha_pool.py         # Factor-pool maintenance and sampling
│   │   └── model.py              # Alpha model interfaces
│   ├── rl/
│   │   ├── env/core.py           # States, actions, and rewards
│   │   ├── env/wrapper.py        # Environment wrapper and action-space refresh
│   │   └── policy.py             # Policy and feature-extractor components
│   └── utils/
│       ├── correlation.py        # Batched correlations and returns with masks
│       ├── pytorch_utils.py      # Daily normalization and masked statistics
│       └── random.py             # Python, PyTorch, and CUDA random seeds
├── alphagen_generic/             # Feature definitions and operator libraries
│   ├── operators.py              # Standard unary, binary, and rolling operators
│   ├── operators_cstm.py         # Custom operators
│   ├── operators_qlib.py         # Qlib operators
│   ├── operators_talib.py        # TA-Lib operators
│   └── features.py               # Shared feature and target definitions
├── alphagen_qlib/                # Qlib adapters
│   ├── calculator.py             # Factor evaluation on Qlib market data
│   └── stock_data.py             # Feature enums and data access
├── gan/
│   ├── dataset/collector.py      # Expression sampling and incremental evaluation
│   ├── network/
│   │   ├── generater.py          # CNN/DCGAN/LSTM generators and training
│   │   ├── masker.py             # Syntax masks for valid token sequences
│   │   ├── predictor.py          # CNN predictors and regression training
│   │   └── loss.py               # Generator loss components and weights
│   └── utils/
│       ├── builder.py            # Expression builders, filtering, and export
│       ├── data.py               # Qlib loading, date splits, and caching
│       ├── qlib.py               # Qlib initialization and instrument access
│       ├── pool.py               # Factor-pool evaluation and weight optimization
│       └── plot_utils.py         # ASCII training and evaluation plots
├── gplearn/                      # Modified genetic programming implementation
│   ├── genetic.py                # Evolution loop
│   ├── functions.py              # Function nodes
│   ├── fitness.py                # Fitness objectives
│   ├── _program.py               # Expression programs and trees
│   └── tests/                    # GP unit tests
├── llm_opt/
│   ├── llm_api.py                # LLM API adapters
│   ├── llm_config.py             # Provider, model, credentials, and run directory
│   ├── llm_loader.py             # Input and reference-document loading
│   ├── llm_main.py               # Optimization entry point
│   ├── llm_parse.py              # Response parsing and JSON repair
│   ├── llm_prmt.py               # English prompts and output requirements
│   └── llm_factor.json           # Factor JSON template
├── backtest/
│   ├── factor_utils.py           # Preprocessing, correlations, returns, and risk metrics
│   ├── factor_analysis.py        # Analysis, plots, and backtest summaries
│   ├── factor_data.py            # Factor-data loading and preparation
│   └── qlib_factor_calculator.py # Qlib-backed expression calculation
├── factor_pipeline/
│   ├── factor_pipeline.py        # Six-stage orchestrator
│   ├── 1_factor_process.py       # Factor series and inventories
│   ├── 2_factor_backtest.py      # Backtests and statistics
│   ├── 3_factor_select.py        # Ranking, selection, and summary tables
│   ├── 4_factor_json.py          # JSON export
│   ├── 5_factor_builder.py       # Builder reconstruction and export
│   └── 6_factor_pool.py          # Integrated factor pools
├── paper_trading/
│   ├── pt_main.py                # Paper-trading entry point
│   ├── pt_pipeline.py            # Pipeline orchestration
│   ├── pt_config.py              # CLI arguments, paths, and configuration
│   ├── pt_calculator.py          # Signal calculation
│   ├── pt_factor.py              # Factor loading
│   ├── pt_mask.py                # Suspension and listing masks
│   ├── pt_stockpool.py           # Stock-pool construction and naming
│   ├── pt_fetcher.py             # Local daily-data loading
│   ├── local_data.py             # Validated local CSV data provider
│   ├── pt_converter.py           # Conversion to Qlib-style data
├── dso/                          # Deep symbolic optimization implementation
│   ├── run.py                    # Multi-seed launcher
│   ├── core.py                   # Model setup and search
│   ├── program.py                # Symbolic programs
│   ├── functions.py              # Symbolic functions
│   ├── library.py                # Function and constant tokens
│   ├── policy/                   # Search policies
│   ├── policy_optimizer/         # Policy optimization
│   ├── task/                     # Task adapters
│   ├── train.py                  # Training loop
│   ├── train_stats.py            # Training statistics
│   └── variance.py               # Variance analysis
├── data_collection/
│   ├── fetch_baostock_data.py     # BaoStock collection and price adjustment
│   └── qlib_dump_bin.py           # CSV-to-Qlib binary conversion
└── md/
    ├── backtest.md               # Backtest metric definitions
    └── operators.md              # Operator signatures and expression rules
```

## Local resources

The following resources are created locally or supplied by the user and are
excluded from the repository:

| Path | Contents |
| --- | --- |
| `analysis/` | Factor series, backtests, selections, and LLM output |
| `out/`, `out_dso/`, `out_gp/`, `out_rl/` | Training and exported factor artifacts |
| `pool/` | Integrated factor pools |
| `pkl/` | Cached Qlib data |
| `.env` | Local credentials and environment settings |

A fresh clone contains only source code, documentation, tests, and dependency
files. Supply market data locally; cached models and generated pools are excluded.
