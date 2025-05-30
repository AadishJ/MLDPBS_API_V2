from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os
import subprocess
from flask_cors import CORS
import gc  

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

    # Handle small datasets to prevent empty train sets
    if len(X) <= 5:
        # For very small datasets, use all data for both training and testing
        return X, X, y, y
    else:
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

            # Determine the start index for future predictions
            if np.array_equal(
                X_train, X_test
            ):  # If we used same data for train and test
                start_idx = len(X_train)
            else:
                start_idx = len(X_train) + len(X_test)

            future_predictions = model.predict(
                np.arange(start_idx, start_idx + future_days).reshape(-1, 1)
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

            # Same logic for future predictions in fallback case
            if np.array_equal(X_train, X_test):
                start_idx = len(X_train)
            else:
                start_idx = len(X_train) + len(X_test)

            future_predictions = model.predict(
                np.arange(start_idx, start_idx + future_days).reshape(-1, 1)
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


def create_batches(df, min_batches=10, max_rows_per_batch=1000):
    """
    Split dataframe into batches with at least min_batches or max_rows_per_batch rows per batch
    """
    total_rows = len(df)

    # Calculate batch size
    if total_rows <= min_batches:
        # Not enough data to create min_batches, so return the full dataset as one batch
        return [df]
    elif total_rows <= max_rows_per_batch * min_batches:
        # We have enough data for min_batches, but not enough for max_rows_per_batch in each
        batch_size = max(1, total_rows // min_batches)
    else:
        # We have lots of data, use max_rows_per_batch
        batch_size = max_rows_per_batch

    # Ensure batch_size is at least 2 to avoid single-row batches
    batch_size = max(2, batch_size)

    # Create batches
    batches = []
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        if end_idx - i > 1:  # Only create batch if it has at least 2 rows
            batches.append(df.iloc[i:end_idx].copy())

    # If no batches were created (edge case), return the full dataset
    if not batches and total_rows > 0:
        batches = [df]

    return batches


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

        datasets = []
        dataset_names = []

        # Process each dataset from the request
        for dataset_obj in data:
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
            df = pd.DataFrame(dataset_data)

            # Validate required columns
            if "Day No." not in df.columns or "Number of entries" not in df.columns:
                return (
                    jsonify(
                        {
                            "error": f"Dataset {name} missing required columns ('Day No.' and 'Number of entries')"
                        }
                    ),
                    400,
                )

            datasets.append(df)
            dataset_names.append(name)

        # Process each dataset separately and calculate percentages for each
        all_dataset_results = []
        all_percentages_for_buddy = []

        for dataset, dataset_name in zip(datasets, dataset_names):
            # Create batches for this dataset
            batches = create_batches(dataset)

            # Process batches for this dataset
            dataset_batch_results = []
            dataset_xgb_averages = []

            for batch_idx, batch in enumerate(batches):
                batch_number = batch_idx + 1

                # Skip empty batches
                if len(batch) == 0:
                    continue

                X_train, X_test, y_train, y_test = prepare_data(batch)
                results = train_and_predict(X_train, X_test, y_train, y_test)

                # Calculate XGBoost average for this batch
                xgb_avg = np.mean(results["XGBoost"]["predictions"])

                dataset_batch_results.append(
                    {
                        "batch_name": f"{dataset_name}_batch_{batch_number}",
                        "batch_number": batch_number,
                        "results": results,
                        "xgb_average": float(xgb_avg),
                        "row_count": len(batch),
                    }
                )

                dataset_xgb_averages.append(xgb_avg)

                # Force garbage collection after each batch
                gc.collect()

            # Calculate percentages for this dataset's batches (should add up to 100%)
            total_sum = sum(dataset_xgb_averages)
            dataset_percentages = [
                (avg / total_sum) * 100 if total_sum > 0 else 0
                for avg in dataset_xgb_averages
            ]

            # Add percentages to batch results
            for i, percentage in enumerate(dataset_percentages):
                dataset_batch_results[i]["percentage"] = float(percentage)

            # Store results for this dataset
            all_dataset_results.append(
                {
                    "dataset_name": dataset_name,
                    "total_rows": len(dataset),
                    "num_batches": len(dataset_batch_results),
                    "batches": dataset_batch_results,
                    "total_percentage_check": sum(
                        dataset_percentages
                    ),  # Should be 100%
                }
            )

            # Collect percentages for buddy allocation
            all_percentages_for_buddy.extend(dataset_percentages)

        # Run BuddyAllocation with all percentages from all datasets
        buddy_output = run_buddy_allocation(all_percentages_for_buddy)

        # Create a flattened list of all batches for easy access
        all_batches_flat = []
        for dataset_result in all_dataset_results:
            all_batches_flat.extend(dataset_result["batches"])

        # Prepare response
        response = {
            "datasets": dataset_names,
            "dataset_results": all_dataset_results,
            "all_batches": all_batches_flat,
            "total_batches": len(all_batches_flat),
            "buddy_allocation_input_percentages": all_percentages_for_buddy,
            "buddy_allocation_output": buddy_output,
        }

        # Force garbage collection before returning
        gc.collect()

        return jsonify(response)

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": trace}), 500


# Add at the very end of your app.py file
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
