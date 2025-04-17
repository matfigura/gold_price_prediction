import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(y_test, y_pred, dates, model_name, save_path=None):
    # Tworzymy DataFrame i sortujemy po dacie
    plot_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_test,
        'Predicted': y_pred
    }).sort_values('Date')

    # Wykres
    plt.figure(figsize=(14, 6))
    plt.plot(plot_df['Date'], plot_df['Actual'], label='Rzeczywiste', linewidth=2, color='blue')
    plt.plot(plot_df['Date'], plot_df['Predicted'], label='Predykcja', linewidth=2, color='orange', linestyle='--')
    plt.title(f"Porównanie wartości rzeczywistych i prognozowanych – {model_name}")
    plt.xlabel("Data")
    plt.ylabel("Cena złota (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()