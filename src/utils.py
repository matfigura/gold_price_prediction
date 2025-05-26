import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.dates as mdates
import math
import seaborn as sns
import numpy as np

def plot_predictions(y_true, y_pred, dates, model_name, save_path):
    plt.figure(figsize=(10, 5))
    sorted_data = sorted(zip(dates, y_true, y_pred), key=lambda x: x[0])
    dates_sorted, y_true_sorted, y_pred_sorted = zip(*sorted_data)
    plt.plot(dates_sorted, y_true_sorted, label="Rzeczywiste", linewidth=2, color='blue')
    plt.plot(dates_sorted, y_pred_sorted, label="Predykcja", linestyle='--', linewidth=2, color='orange')
    plt.xlabel("Data")
    plt.ylabel("Cena zamknięcia")
    plt.title(f"{model_name} – rzeczywiste vs predykcja")
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def join_prediction_plots(folder="results", output_path="results/all_models_preview.png"):
    files = [
        ("decision_tree_plot.png", "Decision Tree"),
        ("random_forest_plot.png", "Random Forest"),
        ("svm_plot.png", "SVM"),
        ("xgboost_plot.png", "XGBoost"),
        ("xgboost_early_plot.png", "XGBoost Early"),
        ("mlp_plot.png", "MLP"),
        ("lstm_plot.png", "LSTM")
    ]

    images = []
    labels = []
    for f, label in files:
        path = os.path.join(folder, f)
        if os.path.exists(path):
            img = Image.open(path).resize((640, 360))
            images.append(img)
            labels.append(label)

    if not images:
        print("[WARN] Nie znaleziono wykresów do połączenia.")
        return

    cols = 3
    rows = math.ceil(len(images) / cols)
    thumb_width, thumb_height = images[0].size
    font = ImageFont.load_default()

    # Zwiększ wysokość na podpisy
    labeled_height = thumb_height + 30
    collage = Image.new('RGB', (cols * thumb_width, rows * labeled_height), (255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * labeled_height
        collage.paste(img, (x, y))

        draw = ImageDraw.Draw(collage)
        bbox = draw.textbbox((0, 0), labels[idx], font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (thumb_width - text_w) // 2, y + thumb_height + 5),
            labels[idx], fill=(0, 0, 0), font=font
        )

    collage.save(output_path)
    print(f"[INFO] Zapisano zbiorczy wykres do: {output_path}")


def plot_error_heatmap(results_df, save_path="results/error_metrics_heatmap.png"):
    # Sprawdzenie wymaganych kolumn
    if not {'model', 'MAE mean', 'RMSE', 'R^2'}.issubset(results_df.columns):
        print("[WARN] Nie znaleziono pełnych danych do heatmapy.")
        return

    # Przygotowanie danych
    df_metrics = results_df[['model', 'MAE mean', 'RMSE', 'R^2']].copy()
    df_metrics.columns = ['Model', 'MAE', 'RMSE', 'R^2']
    df_metrics = df_metrics.sort_values(by='MAE', ascending=False).reset_index(drop=True)
    df_metrics.set_index('Model', inplace=True)

    # Przeskalowanie danych metrycznych (MAE, RMSE do 0–1), R^2 odwrotnie (1–R^2) też do 0–1
    df_scaled = df_metrics.copy()
    df_scaled[['MAE', 'RMSE']] = df_scaled[['MAE', 'RMSE']].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
    df_scaled['R^2'] = 1 - df_scaled['R^2']  # im wyższe R², tym "mniejszy błąd"

    # Rysowanie heatmapy
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df_scaled,
        annot=df_metrics.round(2),
        fmt='',
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Znormalizowana wartość metryki (niższa = lepsza)'}
    )
    plt.title("Heatmapa MAE, RMSE i R² (odwrócony)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()