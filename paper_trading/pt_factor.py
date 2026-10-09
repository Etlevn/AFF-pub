# -*- coding: utf-8 -*-
'''
Step 3: Factor reading module (pt_factor)
'''

import pandas as pd
import os
import pickle
import sys
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from paper_trading.pt_config import PipelineConfig


class PTFactorReader:
    def __init__(self):
        self.top_factor = None
        self.factor_expression = None
        self.factor_score = None
        self.factor_id = None
        self.factor_dir = None
        self.sort_by = None


    def load_top_factor(self, *, config: PipelineConfig):
        factor_dir = config.factor.factor_dir
        factor_name = config.factor.factor_name
        pool_base_dir = config.factor.pool_base_dir
        csv_filename = config.factor.csv_filename
        pkl_filename = config.factor.pkl_filename

        print(f"Factor directory: {factor_dir}")
        print(f"Specify factor name: {factor_name}")

        self.factor_dir = factor_dir
        self.sort_by = None

        try:
            csv_path = os.path.join(project_root, pool_base_dir, factor_dir, csv_filename)
            pkl_dir = os.path.join(project_root, pool_base_dir, factor_dir)
            if not os.path.exists(csv_path):
                print(f"[ERROR] CSV file does not exist: {csv_path}")
                return None
            if not os.path.exists(pkl_dir):
                print(f"[ERROR] The PKL directory does not exist: {pkl_dir}")
                return None
            print("Read CSV file...")
            df = pd.read_csv(csv_path)
            print(f"CSV file contains {len(df)} factors")

            # csv_zoo_final.csv format is fixed, directly rename the column
            df = df.rename(columns={'Unnamed: 0': 'index', 'exprs': 'expression', 'scores': 'score'})
            print("CSV column names have been renamed to match the standard format")

            if not factor_name:
                print(f"[ERROR] factor_name is not specified, please set factor.factor_name in the configuration")
                return None

            def normalize_index(val) -> str:
                s = str(val).strip()
                try:
                    f = float(s)
                    if float(int(f)) == f:
                        return str(int(f))
                    return s
                except Exception:
                    return s

            fn_norm = normalize_index(factor_name)
            df = df.copy()
            df["index_str"] = df["index"].astype(str)
            df["index_norm"] = df["index_str"].apply(normalize_index)

            matched = df[df["index_norm"] == fn_norm]
            if matched.empty:
                print(f"[ERROR] The index is not found in CSV {factor_name} factor")
                print(f"Available indexes: {df['index_str'].tolist()[:10]}...")
                return None

            top_factor_row = matched.iloc[0]
            self.factor_id = str(top_factor_row.get('index', 'N/A'))
            self.factor_score = top_factor_row.get('score', 'N/A')
            csv_expr = top_factor_row.get('expression', '')
            print(f"[INFO] Find the specified factor")
            print(f"Factor ID: {self.factor_id}")
            print(f"Expression string: {csv_expr}")
            print("Read PKL file...")
            pkl_path = os.path.join(pkl_dir, pkl_filename)
            if not os.path.exists(pkl_path):
                print(f"[ERROR] PKL file does not exist: {pkl_path}")
                return None
            with open(pkl_path, 'rb') as f:
                builders_data = pickle.load(f)
            print("Find factors by expression matching...")
            factor_data = None
            factor_index = None
            for i, builder in enumerate(builders_data):
                if builder.exprs_str[0] == csv_expr:
                    factor_data = builder
                    factor_index = i
                    break
            if factor_data is None:
                print(f"[ERROR] No matching expression found in PKL file")
                print(f"CSV expression: {csv_expr}")
                return None
            print(f"[INFO] Find the matching factor, PKL index: {factor_index}")
            if hasattr(factor_data, 'exprs_str'):
                self.factor_expression = factor_data.exprs_str
            elif hasattr(factor_data, 'expression'):
                self.factor_expression = factor_data.expression
            elif isinstance(factor_data, dict) and 'expression' in factor_data:
                self.factor_expression = factor_data['expression']
            else:
                print("[ERROR] Unable to extract factor expression from PKL file")
                print(f"Available attributes: {[attr for attr in dir(factor_data) if not attr.startswith('_')]}")
                return None
            self.top_factor = {
                'id': self.factor_id,
                'score': self.factor_score,
                'expression': self.factor_expression,
                'expr_str': csv_expr,
                'all_data': top_factor_row.to_dict(),
                'factor_data': factor_data,
                'pkl_index': factor_index
            }
            print(f"[INFO] Successfully loaded factor expression")
            print(f"Expression length: {len(self.factor_expression)}")
            print(f"\n[DONE] factor read successfully")
            return self.get_factor_info()
        except Exception as e:
            print(f"[ERROR] Loading factor failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_factor_info(self):
        if self.top_factor is None:
            return None
        return {
            'id': self.factor_id,
            'expression': self.factor_expression,
            'score': self.factor_score,
            'expr_str': self.top_factor.get('expr_str', ''),
            'all_data': self.top_factor['all_data'],
            'factor_data': self.top_factor.get('factor_data', None)
        }
