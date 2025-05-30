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

        # Create batches for each dataset
        all_dataset_batches = []
        max_batches = 0

        for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
            batches = create_batches(dataset)
            all_dataset_batches.append({"dataset_name": name, "batches": batches})
            max_batches = max(max_batches, len(batches))

        # Process batches and make predictions
        batch_predictions = []

        # Group batches by batch number (batch_1, batch_2, etc.)
        batch_groups = {}

        for dataset_info in all_dataset_batches:
            dataset_name = dataset_info["dataset_name"]
            batches = dataset_info["batches"]

            for batch_idx, batch in enumerate(batches):
                batch_number = batch_idx + 1
                batch_key = f"batch_{batch_number}"

                if batch_key not in batch_groups:
                    batch_groups[batch_key] = []

                # Skip empty batches
                if len(batch) == 0:
                    continue

                X_train, X_test, y_train, y_test = prepare_data(batch)
                results = train_and_predict(X_train, X_test, y_train, y_test)

                # Calculate XGBoost average for this batch
                xgb_avg = np.mean(results["XGBoost"]["predictions"])

                batch_groups[batch_key].append(
                    {
                        "dataset_name": dataset_name,
                        "batch_name": f"{dataset_name}_batch_{batch_number}",
                        "results": results,
                        "xgb_average": float(xgb_avg),
                        "row_count": len(batch),
                    }
                )

                # Force garbage collection after each batch
                gc.collect()

        # Calculate combined XGBoost averages for each batch group
        combined_batch_results = []
        batch_group_averages = []

        for batch_key in sorted(
            batch_groups.keys(), key=lambda x: int(x.split("_")[1])
        ):
            batch_data = batch_groups[batch_key]

            # Calculate weighted average across all datasets for this batch number
            total_weighted_sum = 0.0
            total_rows = sum(item["row_count"] for item in batch_data)

            for item in batch_data:
                weight = item["row_count"] / total_rows if total_rows > 0 else 0
                total_weighted_sum += item["xgb_average"] * weight

            combined_batch_results.append(
                {
                    "batch_group": batch_key,
                    "datasets_in_batch": [item["dataset_name"] for item in batch_data],
                    "combined_xgb_average": float(total_weighted_sum),
                    "total_rows": total_rows,
                    "individual_batches": batch_data,
                }
            )

            batch_group_averages.append(total_weighted_sum)

        # Calculate percentages for batch groups (should add up to 100%)
        total_sum = sum(batch_group_averages)
        batch_group_percentages = [
            (avg / total_sum) * 100 if total_sum > 0 else 0
            for avg in batch_group_averages
        ]

        # Add percentages to combined batch results
        for i, percentage in enumerate(batch_group_percentages):
            combined_batch_results[i]["percentage"] = float(percentage)

        # Run BuddyAllocation with batch group percentages
        buddy_output = run_buddy_allocation(batch_group_percentages)

        # Create dataset summary showing which batches each dataset contributes to
        dataset_summary = {}
        for dataset_name in dataset_names:
            dataset_summary[dataset_name] = {"total_rows": 0, "batch_contributions": []}

        # Fill dataset summary
        for batch_result in combined_batch_results:
            for individual_batch in batch_result["individual_batches"]:
                dataset_name = individual_batch["dataset_name"]
                dataset_summary[dataset_name]["total_rows"] += individual_batch[
                    "row_count"
                ]
                dataset_summary[dataset_name]["batch_contributions"].append(
                    {
                        "batch_group": batch_result["batch_group"],
                        "row_count": individual_batch["row_count"],
                        "xgb_average": individual_batch["xgb_average"],
                    }
                )

        # Prepare response
        response = {
            "datasets": dataset_names,
            "batch_groups": combined_batch_results,
            "dataset_summary": dataset_summary,
            "allocation_summary": {
                batch_result["batch_group"]: batch_result["percentage"]
                for batch_result in combined_batch_results
            },
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
