import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_preprocessing import prepare_data

# 1️⃣ Wczytaj dane tak, by w feature_names znalazły się wszystkie cechy pochodne i techniczne,
#    bez surowego OHLC:
X_train, _, _, _, _, feature_names = prepare_data(
    file_path="data/gold_data.csv",
    include_ohlc=True,
    include_features=True,
    include_technical_indicators=True,
    drop_raw_ohlc_after_feature_gen=True,
    filter_correlated_features=False,
    filter_low_target_correlation=False
)

# 2️⃣ Zamień je na DataFrame:
df_all = pd.DataFrame(X_train, columns=feature_names)

# 3️⃣ Zdefiniuj listę ważnych cech:
selected_features = [
    "mid_price_1",
    "ema_12",
    "macd",
    "obv",
    "Volume_lag3",
    "Volume_lag2",
    "Volume_lag1",
    "Volume"
]

# 4️⃣ Zweryfikuj, które z tych nazw faktycznie są w feature_names:
available = [f for f in selected_features if f in feature_names]
missing   = [f for f in selected_features if f not in feature_names]

print("Znalezione w feature_names:", available)
if missing:
    print("Brakujące (nie występują w feature_names):", missing)

# 5️⃣ Stwórz pod‐DataFrame tylko z dostępnych kolumn i usuń ewentualne NaN:
df_sel = df_all[available].dropna()

# 6️⃣ Oblicz macierz korelacji:
corr = df_sel.corr()

# 7️⃣ Rysuj heatmapę:
plt.figure(
    figsize=(
        max(8, len(corr.columns) * 0.5),
        max(6, len(corr.columns) * 0.5)
    )
)
sns.heatmap(corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1)

plt.title("Heatmapa korelacji wybranych cech")
plt.tight_layout()
plt.show()