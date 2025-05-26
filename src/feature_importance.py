import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance_comparison(features, dt_importance, rf_importance, save_path='results/feature_importance_comparison.png'):
    # Tworzenie DataFrame do porównania
    importance_df = pd.DataFrame({
        'Cecha': features,
        'Drzewo decyzyjne': dt_importance,
        'Las losowy': rf_importance
    })

    # Sortowanie względem średniej ważności
    importance_df['Średnia'] = (importance_df['Drzewo decyzyjne'] + importance_df['Las losowy']) / 2
    importance_df = importance_df.sort_values(by='Średnia', ascending=True)

    # Wykres słupkowy poziomy
    ax = importance_df.plot(
        x='Cecha',
        y=['Drzewo decyzyjne', 'Las losowy'],
        kind='barh',
        figsize=(8, 5),
        title='Porównanie ważności cech – Drzewo vs Las losowy'
    )
    plt.xlabel('Ważność cechy')
    plt.tight_layout()
    plt.show()
