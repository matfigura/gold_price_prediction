from src.data_preprocessing import prepare_data
from src.models.decision_tree import train_decision_tree, predict_decision_tree
from src.models.random_forest import train_random_forest, predict_random_forest
from src.models.svm import train_svm, predict_svm
from src.utils import plot_predictions
import pandas as pd

from src.data_preprocessing import prepare_data
from src.models.decision_tree import train_decision_tree, predict_decision_tree
from src.utils import plot_predictions
import pandas as pd

X_train, X_test, y_train, y_test, dates_test = prepare_data()
results = [] 

# Trening z GridSearchCV i pobranie najlepszego modelu
result_dt, best_model_dt = train_decision_tree(X_train, y_train)
results.append(result_dt)
# Predykcja i wykres
y_pred_dt = predict_decision_tree(best_model_dt, X_test)
plot_predictions(y_test, y_pred_dt, dates_test, model_name="Decision Tree", save_path="results/decision_tree_plot.png")

# Random Forest
result_rf, best_model_rf = train_random_forest(X_train, y_train)
results.append(result_rf)
y_pred_rf = predict_random_forest(best_model_rf, X_test)
plot_predictions(y_test, y_pred_rf, dates_test, model_name="Random Forest", save_path="results/random_forest_plot.png")


# SVM (SVR)
result_svm, best_model_svm = train_svm(X_train, y_train)
results.append(result_svm)
y_pred_svm = predict_svm(best_model_svm, X_test)
plot_predictions(y_test, y_pred_svm, dates_test, model_name="SVM", save_path="results/svm_plot.png")

results_df = pd.DataFrame(results)
results_df.to_csv('results/comparison_table.csv', index=False)
print(results_df)
