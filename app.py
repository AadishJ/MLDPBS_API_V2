from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os
import subprocess
import tempfile
import json
from flask_cors import CORS
import gc  # Import garbage collector
import math

app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(
    app,
    resources={
        r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]}
    },
)

# Set up gunicorn timeout configuration
app.config["TIMEOUT"] = 300  # 5 minutes instead of default 30 seconds


def prepare_data(data):
    X = data["Day No."].values.reshape(-1, 1)
    y = data["Number of entries"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_predict(X_train, X_test, y_train, y_test, future_days=7):
    # Use only two models instead of four to save memory
    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "GBR": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        try:
            # Set lower complexity for models to save memory
            if name == "XGBoost":
                model.set_params(max_depth=3, n_estimators=50)
            elif name == "GBR":
                model.set_params(max_depth=3, n_estimators=50)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            future_predictions = model.predict(
                np.arange(
                    len(X_train) + len(X_test), len(X_train) + len(X_test) + future_days
                ).reshape(-1, 1)
            )

            # Explicitly clean up memory
            gc.collect()

        except Exception as e:
            print(f"Error training {name} model: {str(e)}")
            print(f"Falling back to LinearRegression for {name}")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            future_predictions = model.predict(
                np.arange(
                    len(X_train) + len(X_test), len(X_train) + len(X_test) + future_days
                ).reshape(-1, 1)
            )

        results[name] = {"mse": float(mse), "predictions": future_predictions.tolist()}

    return results


def run_buddy_allocation(percentages):
    """Run the BuddyAllocation program with the given percentages."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buddy_executable = os.path.join(script_dir, "BuddyAllocation")

    # Create a predictions directory if it doesn't exist
    predictions_dir = os.path.join(script_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Create the percentages file
    percentages_file = os.path.join(predictions_dir, "percentages.txt")
    with open(percentages_file, "w") as f:
        for percentage in percentages:
            f.write(f"{percentage:.2f}\n")

    try:
        # Check if the executable exists
        if os.path.exists(buddy_executable):
            # Run the executable with a timeout to prevent hanging
            run_result = subprocess.run(
                [buddy_executable],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Just return the results section of the output
            output_lines = run_result.stdout.split("\n")
            result_section = []
            results_found = False

            for line in output_lines:
                if line.strip() == "Results:":
                    results_found = True
                    result_section.append(line)
                elif results_found:
                    result_section.append(line)

            return "\n".join(result_section)
        else:
            return "BuddyAllocation executable not found. It should be compiled during the build process."
    except subprocess.TimeoutExpired:
        return "BuddyAllocation execution timed out after 30 seconds."
    except subprocess.CalledProcessError as e:
        return f"Error running BuddyAllocation: {e.stderr}"


def slice_dataframe(df, max_slices=10, max_rows_per_slice=1000):
    """
    Slice a dataframe into chunks, with a maximum of max_slices or
    chunks of max_rows_per_slice rows, whichever results in smaller chunks.
    """
    total_rows = len(df)

    # Calculate rows per slice based on max_slices
    rows_per_slice_by_count = math.ceil(total_rows / max_slices)

    # Choose the smaller slice size between the two options
    rows_per_slice = min(rows_per_slice_by_count, max_rows_per_slice)

    # Calculate the actual number of slices
    num_slices = math.ceil(total_rows / rows_per_slice)

    slices = []
    for i in range(num_slices):
        start_idx = i * rows_per_slice
        end_idx = min((i + 1) * rows_per_slice, total_rows)
        slice_df = df.iloc[start_idx:end_idx].copy()
        slices.append(slice_df)

    return slices


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        if not data or not isinstance(data, list):
            return (
                jsonify(
                    {
                        "error": "Invalid data format. Expected an array of dataset objects"
                    }
                ),
                400,
            )

        # Limit number of datasets to prevent memory issues
        if len(data) > 5:
            return (
                jsonify(
                    {"error": "Too many datasets. Maximum 5 allowed for processing."}
                ),
                400,
            )

        all_results = []
        dataset_names = []
        slice_averages = []

        # Process each dataset from the request
        for dataset_index, dataset_obj in enumerate(data):
            if "name" not in dataset_obj or "data" not in dataset_obj:
                return (
                    jsonify(
                        {"error": "Each dataset must have 'name' and 'data' fields"}
                    ),
                    400,
                )

            name = dataset_obj["name"]
            dataset_data = dataset_obj["data"]

            # Convert to pandas DataFrame
            original_df = pd.DataFrame(dataset_data)

            # Validate required columns
            if (
                "Day No." not in original_df.columns
                or "Number of entries" not in original_df.columns
            ):
                return (
                    jsonify(
                        {
                            "error": f"Dataset {name} missing required columns ('Day No.' and 'Number of entries')"
                        }
                    ),
                    400,
                )

            # Slice the dataset
            df_slices = slice_dataframe(
                original_df, max_slices=10, max_rows_per_slice=1000
            )

            # Process each slice
            dataset_results = []
            slice_xgb_avgs = []

            for slice_index, slice_df in enumerate(df_slices):
                slice_name = f"{name}_slice_{slice_index+1}"

                # Process this slice
                X_train, X_test, y_train, y_test = prepare_data(slice_df)
                slice_result = train_and_predict(X_train, X_test, y_train, y_test)

                # Store results for this slice
                dataset_results.append(
                    {
                        "slice": slice_index + 1,
                        "row_count": len(slice_df),
                        "results": slice_result,
                    }
                )

                # Store XGBoost average for this slice
                slice_xgb_avg = np.mean(slice_result["XGBoost"]["predictions"])
                slice_xgb_avgs.append(slice_xgb_avg)

                # Force garbage collection after each slice
                gc.collect()

            # Calculate overall average for this dataset
            dataset_xgb_avg = np.mean(slice_xgb_avgs)

            # Store results for this dataset
            all_results.append(
                {
                    "dataset_name": name,
                    "slice_count": len(df_slices),
                    "total_rows": len(original_df),
                    "slice_results": dataset_results,
                    "xgb_average": float(dataset_xgb_avg),
                }
            )

            dataset_names.append(name)
            slice_averages.append(dataset_xgb_avg)

            # Force garbage collection after each dataset
            gc.collect()

        # Calculate percentages
        total_sum = sum(slice_averages)
        percentages = [
            (avg / total_sum) * 100 if total_sum > 0 else 0 for avg in slice_averages
        ]

        # Run BuddyAllocation if available
        buddy_output = run_buddy_allocation(percentages)

        # Prepare response
        response = {
            "datasets": dataset_names,
            "predictions": [
                {
                    "dataset": name,
                    "xgb_average": float(xgb_avg),
                    "percentage": float(pct),
                    "detailed_results": result,
                }
                for name, result, xgb_avg, pct in zip(
                    dataset_names, all_results, slice_averages, percentages
                )
            ],
            "buddy_allocation_output": buddy_output,
        }

        # Force garbage collection before returning
        gc.collect()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add at the very end of your app.py file
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
