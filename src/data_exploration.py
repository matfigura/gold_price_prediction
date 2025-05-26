import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(file_path='data/gold_data.csv', save_path='results/correlation_heatmap.png'):
    # Wczytanie danych z separatorem średnika
    df = pd.read_csv(file_path, sep=';')

    # Konwersja kolumny 'Date' na datę (niepotrzebna do korelacji)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.drop(columns=['Date'])

    # Obliczenie macierzy korelacji
    correlation_matrix = df.corr(method='pearson')

    # Ustawienia wykresu
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Heatmapa korelacji zmiennych')
    plt.tight_layout()

    # Zapis lub wyświetlenie wykresu
    plt.savefig(save_path)
    plt.show()

    