import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import matplotlib.dates as mdates
import math


def plot_predictions(y_true, y_pred, dates, model_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Rzeczywiste", linewidth=2, color='blue')
    plt.plot(dates, y_pred, label="Predykcja", linestyle='--', linewidth=2, color='orange')
    plt.xlabel("Data")
    plt.ylabel("Cena zamknięcia")
    plt.title(f"{model_name} – rzeczywiste vs predykcja")
    plt.legend()
    plt.grid(True)

    # Formatowanie daty
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def join_prediction_plots(folder="results", output_path="results/all_models_preview.png"):
    files = [
        "decision_tree_plot.png",
        "random_forest_plot.png",
        "svm_plot.png",
        "xgboost_plot.png",
        "xgboost_early_plot.png",
        "mlp_plot.png",
        "lstm_plot.png"
    ]

    images = [Image.open(os.path.join(folder, f)).resize((640, 360)) for f in files if os.path.exists(os.path.join(folder, f))]

    if not images:
        print("[WARN] Nie znaleziono wykresów do połączenia.")
        return

    cols = 3
    rows = math.ceil(len(images) / cols)
    thumb_width, thumb_height = images[0].size
    collage_width = cols * thumb_width
    collage_height = rows * thumb_height

    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    for idx, im in enumerate(images):
        x_offset = (idx % cols) * thumb_width
        y_offset = (idx // cols) * thumb_height
        collage.paste(im, (x_offset, y_offset))

    collage.save(output_path)
    print(f"[INFO] Zapisano zbiorczy wykres do: {output_path}")