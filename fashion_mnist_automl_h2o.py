# Import necessary libraries
import numpy as np
import pandas as pd
import sweetviz as sv
import shap
import h2o
from h2o.automl import H2OAutoML
import mlflow
from scipy.stats import ks_2samp
from tensorflow.keras.datasets import fashion_mnist # type: ignore
# from pandas_profiling import ProfileReport # type: ignore
from ydata_profiling import ProfileReport

# M1: Exploratory Data Analysis (EDA)
# def perform_eda():
#     # Load Fashion MNIST dataset
#     (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#     # Flatten the image data for EDA
#     X_train_flat = X_train.reshape(X_train.shape[0], -1)
#     df_train = pd.DataFrame(X_train_flat)
#     df_train['label'] = y_train

#     # Use a smaller subset for EDA
#     df_train_sample = df_train.sample(frac=0.1, random_state=42)  # 10% of the data

#     # Generate EDA report using ydata-profiling
#     profile = ProfileReport(df_train_sample, title="Fashion MNIST EDA Report")
#     profile.to_file("fashion_mnist_eda.html")

#     print("EDA report generated: fashion_mnist_eda.html")
#     return X_train, y_train, X_test, y_test

def perform_eda():
    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Resize images to 14x14 to reduce the number of features
    from skimage.transform import resize
    X_train_resized = np.array([resize(img, (14, 14)) for img in X_train])
    X_train_flat = X_train_resized.reshape(X_train_resized.shape[0], -1)
    df_train = pd.DataFrame(X_train_flat)
    df_train['label'] = y_train

    # Use a smaller subset for EDA
    df_train_sample = df_train.sample(frac=0.1, random_state=42)  # 10% of the data

    # Generate EDA report using ydata-profiling (minimal mode)
    profile = ProfileReport(df_train_sample, title="Fashion MNIST EDA Report", minimal=True)
    profile.to_file("fashion_mnist_eda.html")

    print("EDA report generated: fashion_mnist_eda.html")
    return X_train, y_train, X_test, y_test

# M2: Feature Engineering & Explainability
def feature_engineering_and_explainability(X_train, y_train):
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0

    # Use SHAP for explainability
    sample_idx = np.random.choice(X_train.shape[0], 1000, replace=False)
    X_sample = X_train[sample_idx].reshape(-1, 28 * 28)

    # Train a simple model for SHAP explainability
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_sample, y_train[sample_idx])

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Visualize SHAP summary
    shap.summary_plot(shap_values, X_sample, feature_names=[f'pixel_{i}' for i in range(28 * 28)])
    print("SHAP explainability visualization completed.")

    return X_train

# M3: Model Selection & Hyperparameter Optimization
def model_selection_and_optimization(X_train, y_train):
    # Initialize H2O
    h2o.init()

    # Convert data to H2O frame
    df_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    df_train['label'] = y_train
    train_df = h2o.H2OFrame(df_train)
    train_df['label'] = train_df['label'].asfactor()

    # Run H2O AutoML
    aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=300)
    aml.train(x=train_df.columns[:-1], y='label', training_frame=train_df)

    # View leaderboard
    lb = aml.leaderboard
    print(lb)

    # Save the leader model
    best_model = aml.leader
    h2o.save_model(best_model, path="best_model")

    print("Model selection and optimization completed. Best model saved.")
    return best_model

# M4: Model Monitoring & Performance Tracking
def model_monitoring_and_tracking(X_train, X_test, best_model):
    # Log metrics and artifacts using MLflow
    mlflow.start_run()
    mlflow.log_metric('accuracy', best_model.accuracy())
    mlflow.log_artifact('fashion_mnist_eda.html')
    mlflow.end_run()

    # Drift detection using Kolmogorov-Smirnov test
    drift_results = ks_2samp(X_train.flatten(), X_test.flatten())
    print(f"Drift p-value: {drift_results.pvalue}")

    print("Model monitoring and drift detection completed.")

# M5: Final Deliverables
def final_deliverables():
    # Zip file containing code, processed data, and trained model
    import shutil
    shutil.make_archive('assignment2_deliverables', 'zip', '.')

    print("Final deliverables zipped: assignment2_deliverables.zip")

# Main function to execute all milestones
def main():
    # M1: EDA
    X_train, y_train, X_test, y_test = perform_eda()

    # M2: Feature Engineering & Explainability
    X_train = feature_engineering_and_explainability(X_train, y_train)

    # M3: Model Selection & Hyperparameter Optimization
    best_model = model_selection_and_optimization(X_train, y_train)

    # M4: Model Monitoring & Performance Tracking
    model_monitoring_and_tracking(X_train, X_test, best_model)

    # M5: Final Deliverables
    final_deliverables()

if __name__ == "__main__":
    main()