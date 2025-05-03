import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(df_evaluate, y_true, y_pred, model_name: str, with_gd: bool):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    row = {
        "Model": model_name,
        "Z gospodarstwami domowymi": with_gd,
        "R-Squared": round(r2, 2),
        "Mean Absolute Error (MAE)": round(mae, 2),
        "Root Mean Squared Error (RMSE)": round(rmse, 2),

    }
    df_evaluate = pd.concat([df_evaluate, pd.DataFrame([row])])

    return df_evaluate


def plot_features_importance(features, feature_importance, model: str):
    # Plot feature importance
    plt.barh(features, feature_importance)
    plt.xlabel("Feature Importance")
    plt.yticks(rotation=25)
    plt.title(f"{model} - Istotność cech")
    # Add values at the end of the bars
    for index, value in enumerate(feature_importance):
        plt.text(value, index, f'{value:.2f}')  # Adjust format as needed

    plt.show()