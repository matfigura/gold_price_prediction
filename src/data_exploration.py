import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

  

def plot_correlation_heatmap(file_path='data/gold_data.csv', save_path='results/correlation_heatmap.png'):
    # Wczytanie danych z separatorem średnika
    df = pd.read_csv(file_path, sep=';')

    # Konwersja kolumny daty i sortowanie
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        df = df.drop(columns=['Date'])


    # Usunięcie wierszy z brakami danych (po przesunięciach lagowych)
    df = df.dropna()

    # Wyświetlenie informacji diagnostycznych
    print("\nOpis statystyczny wybranych kolumn cenowych:")
    print(df[['Open', 'High', 'Low', 'Close']].describe())

    print("\nPrzykładowe wartości (pierwsze 20 wierszy):")
    print(df[['Open', 'High', 'Low', 'Close']].head(20))

    print("\nMaksymalne różnice między kolumnami (czy są identyczne?):")
    print("Open vs Close:", (df['Open'] - df['Close']).abs().max())
    print("High vs Low:", (df['High'] - df['Low']).abs().max())

    # Zostawiamy tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Obliczenie macierzy korelacji
    corr = numeric_df.corr()

    # Rysowanie heatmapy
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.5,
        fmt=".2f"
    )
    plt.title('Heatmapa korelacji zmiennych OHLC + cechy pochodne')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()