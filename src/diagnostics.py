from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def plot_learning_curve_dt(model, X, y, cv=5, scoring='r2', save_path='results/learning_curve_dt.png'):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Train', marker='o')
    plt.plot(train_sizes, val_scores_mean, label='Validation', marker='s')
    plt.xlabel('Rozmiar zbioru treningowego')
    plt.ylabel(scoring.upper())
    plt.title('Krzywa uczenia – Decision Tree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_validation_curve_dt(X, y, param_name, param_range, save_path):
    model = DecisionTreeRegressor(random_state=42)
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5, scoring='r2', n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label="Train", marker='o')
    plt.plot(param_range, val_mean, label="Validation", marker='s')
    plt.xlabel(param_name)
    plt.ylabel("R²")
    plt.title(f'Krzywa złożoności – {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()