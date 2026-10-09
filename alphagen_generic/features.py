from alphagen.data.expression import Feature, Ref, Log
from alphagen_qlib.stock_data import FeatureType


high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)

# Target variable definition - 1 daily rate of return (reduce missing data)
target = Ref(close, -1) / close - 1

# Improved target definition options
# 1. Shorter-term yield, reducing data missing
target_1d = Ref(close, -1) / close - 1  # 1 daily rate of return
target_3d = Ref(close, -3) / close - 1  # 3 daily rate of return

# 2. Use VWAP as a more stable price benchmark
target_vwap_1d = Ref(vwap, -1) / vwap - 1  # VWAP 1 daily yield
target_vwap_3d = Ref(vwap, -3) / vwap - 1  # VWAP 3 daily yield

# 3. Logarithmic rate of return, more stable
target_log_1d = Log(Ref(close, -1)) - Log(close)  # Logarithmic 1 daily rate of return
target_log_3d = Log(Ref(close, -3)) - Log(close)  # Logarithmic 3 daily rate of return

# 4. Price Momentum Indicator
target_momentum_5d = (close - Ref(close, -5)) / Ref(close, -5)  # 5 daily momentum
target_momentum_10d = (close - Ref(close, -10)) / Ref(close, -10)  # 10 daily momentum

# 5. Relative Strength Index
target_rs_5d = close / Ref(close, -5) - 1  # 5 daily relative strength
target_rs_10d = close / Ref(close, -10) - 1  # 10 daily relative strength

# Recommended target (can be selected according to data quality)
# If the data quality is better, use the original target
# If the data is sparse, use the shorter-term target
target_robust = target_1d  # 1 daily yield, the least missing data
