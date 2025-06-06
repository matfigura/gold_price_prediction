# ———————— te trzy zestawy cech muszą być na poziomie modułu ————————
features_1 = [
    "mid_price", "Volume", "wrobv", "rsi", "macd", "stoch_k",
    "atr", "adx", "cci", "fib_pct", "ichimoku_kijun", "bb_width"
]

features_2 = [
    "mid_price", "Volume", "wrobv", "rsi", "bb_width", "adx", "stoch_k"
]

features_3 = [
    "mid_price", "Volume", "wrobv", "rsi", "bb_width"
]
# ————————————————————————————————————————————————————————————————

from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    output_dir = "results/analysis_feature"
    os.makedirs(output_dir, exist_ok=True)

    from src.data_preprocessing import prepare_data
    from src.data_preprocessing import add_technical_indicators

    X_train, X_test, y_train, y_test, dates_test, feature_names = prepare_data(
        file_path="data/gold_data.csv",
        include_midprice=True,
        include_ohlc=False,
        include_technical_indicators=True
    )

    print("\n>> Ostateczne kolumny używane w X:")
    for col in feature_names:
        print("  -", col)

    df = pd.read_csv("data/gold_data.csv", sep=';')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['target'] = df['Close'].shift(-1)
    df['mid_price'] = (df['High'] + df['Low']) / 2
    df = add_technical_indicators(df)
    df = df.drop(columns=['Open', 'High', 'Low', 'Close'], errors='ignore')
    df = df.dropna(subset=['target']).dropna()

    df_features = df[feature_names].copy()

    print("\nWymiary DataFrame z cechami (przed skalowaniem):", df_features.shape)
    print("Przykład pierwszych 6 kolumn:\n")
    print(df_features.iloc[:, :6].head(), "\n")

    corr_matrix_all = df_features.corr()
    plt.figure(figsize=(16, 14), dpi=150)
    sns.heatmap(corr_matrix_all, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.title("Macierz korelacji (Pearson) – Zmian OHLC na mid_price", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_all_features.png"))
    plt.show()

    X = df_features.values
    cols = df_features.columns
    vif_data = [(cols[i], variance_inflation_factor(X, i)) for i in range(X.shape[1])]
    vif_df_all = pd.DataFrame(vif_data, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
    print("\nTop 20 cech według VIF (wszystkie cechy):\n", vif_df_all.head(20))
    vif_df_all.to_csv(os.path.join(output_dir, "vif_all_features.csv"), index=False)

    # ——— teraz wykorzystujemy module-level lists ———
    missing = [f for f in features_1 if f not in df_features.columns]
    if missing:
        raise ValueError(f"Poniższe cechy nie występują w df_features: {missing}")

    df_selected = df_features[features_1].copy()
    print("\nWymiary DataFrame z finalnymi cechami:", df_selected.shape)

    corr_matrix_sel = df_selected.corr()
    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(corr_matrix_sel, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.title("Macierz korelacji (Pearson) – features_1", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_features_1.png"))
    plt.show()

    X_sel = df_selected.values
    cols_sel = df_selected.columns
    vif_data_sel = [(cols_sel[i], variance_inflation_factor(X_sel, i)) for i in range(X_sel.shape[1])]
    vif_df_sel = pd.DataFrame(vif_data_sel, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
    print("\nTop cech według VIF (finalne cechy):\n", vif_df_sel)
    vif_df_sel.to_csv(os.path.join(output_dir, "vif_features_1.csv"), index=False)

    missing2 = [f for f in features_2 if f not in df_features.columns]
    if missing2:
        raise ValueError(f"Poniższe cechy nie występują w df_features: {missing2}")

    df_super = df_features[features_2].copy()
    print("\nWymiary DataFrame z features_2:", df_super.shape)

    corr_matrix_super = df_super.corr()
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(corr_matrix_super, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.title("Macierz korelacji – features_2", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_features_2.png"))
    plt.show()

    X_super = df_super.values
    cols_super = df_super.columns
    vif_data_super = [(cols_super[i], variance_inflation_factor(X_super, i)) for i in range(X_super.shape[1])]
    vif_df_super = pd.DataFrame(vif_data_super, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
    print("\nTop cech według VIF features_2 :\n", vif_df_super)
    vif_df_super.to_csv(os.path.join(output_dir, "vif_features_2.csv"), index=False)

    missing_ult = [f for f in features_3 if f not in df_features.columns]
    if missing_ult:
        raise ValueError(f"Poniższe cechy nie występują w df_features: {missing_ult}")

    df_ult = df_features[features_3].copy()
    print("Wymiary DataFrame z features_3:", df_ult.shape)

    corr_matrix_ult = df_ult.corr()
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(corr_matrix_ult, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.title("Macierz korelacji – features_3", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_features_3.png"))
    plt.show()

    X_ult = df_ult.values
    cols_ult = df_ult.columns
    vif_data_ult = [(cols_ult[i], variance_inflation_factor(X_ult, i)) for i in range(X_ult.shape[1])]
    vif_df_ult = pd.DataFrame(vif_data_ult, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
    print("\nVIF dla features_3:\n", vif_df_ult)
    vif_df_ult.to_csv(os.path.join(output_dir, "vif_features_3.csv"), index=False)

if __name__ == "__main__":
    main()