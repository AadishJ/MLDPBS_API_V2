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


def run_buddy_allocation(percentages, entry_sizes, batch_number=1):
    """Run the BuddyAllocation program with the given percentages and entry sizes."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buddy_executable = os.path.join(script_dir, "BuddyAllocation")

    # Create a predictions directory if it doesn't exist
    predictions_dir = os.path.join(script_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Create the percentages file with both percentages and block sizes
    percentages_file = os.path.join(predictions_dir, f"percentages_batch_{batch_number}.txt")
    with open(percentages_file, "w") as f:
        # Write percentages first
        for percentage in percentages:
            f.write(f"{percentage:.2f}\n")
        
        # Write block sizes (entry sizes) second
        for entry_size in entry_sizes:
            f.write(f"{entry_size}\n")

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


def calculate_batch_sizes(predictions, min_batch=10, max_batch=100):
    """Calculate appropriate batch sizes based on predictions."""
    batch_sizes = []
    
    for pred_avg in predictions:
        # Use the prediction average to determine batch size
        # Scale between min and max batch sizes
        normalized_pred = max(1, min(pred_avg, 1000))  # Clamp between 1 and 1000
        
        # Calculate batch size: higher predictions get larger batches
        batch_size = int(min_batch + (normalized_pred / 1000) * (max_batch - min_batch))
        batch_size = max(min_batch, min(batch_size, max_batch))
        
        batch_sizes.append(batch_size)
    
    return batch_sizes

def create_batches_from_datasets(datasets, dataset_names, entry_sizes, min_batch_size=10, max_batches=20):
    """Create batches by taking entries from each dataset."""
    all_batches = []
    
    # Calculate the maximum number of batches we can create
    max_possible_batches = min([len(dataset) // min_batch_size for dataset in datasets])
    max_possible_batches = max(1, max_possible_batches)  # At least 1 batch
    
    # Limit to max_batches
    num_batches_to_create = min(max_batches, max_possible_batches)
    
    print(f"Will create {num_batches_to_create} batches")
    
    for batch_idx in range(num_batches_to_create):
        # For each batch, take min_batch_size entries from each dataset
        batch_data_combined = []
        batch_entry_sizes = []
        batch_dataset_names = []
        
        for dataset_idx, (dataset, name, entry_size) in enumerate(zip(datasets, dataset_names, entry_sizes)):
            start_idx = batch_idx * min_batch_size
            end_idx = min(start_idx + min_batch_size, len(dataset))
            
            if start_idx < len(dataset):
                # Take entries from this dataset for this batch
                batch_entries = dataset.iloc[start_idx:end_idx].copy()
                
                # Add dataset identifier to distinguish entries
                batch_entries['source_dataset'] = name
                batch_entries['source_entry_size'] = entry_size
                
                batch_data_combined.append(batch_entries)
                batch_entry_sizes.append(entry_size)
                batch_dataset_names.append(name)
        
        if batch_data_combined:
            # Combine all dataset entries for this batch
            combined_df = pd.concat(batch_data_combined, ignore_index=True)
            
            # Create batch info
            batch_info = {
                'data': combined_df,
                'name': f"batch_{batch_idx + 1}",
                'dataset_names': batch_dataset_names,
                'entry_sizes': batch_entry_sizes,
                'batch_number': batch_idx + 1,
                'row_count': len(combined_df),
                'datasets_in_batch': len(batch_dataset_names)
            }
            
            all_batches.append(batch_info)
    
    return all_batches


def process_batch(batch_info):
    """Process a single batch treating each dataset separately."""
    try:
        data = batch_info['data']
        dataset_names = batch_info['dataset_names']
        entry_sizes = batch_info['entry_sizes']
        
        # Process each dataset within the batch separately
        dataset_results = []
        for i, (dataset_name, entry_size) in enumerate(zip(dataset_names, entry_sizes)):
            # Filter data for this specific dataset
            dataset_data = data[data['source_dataset'] == dataset_name].copy()
            
            # Remove the helper columns for ML processing
            dataset_data = dataset_data.drop(['source_dataset', 'source_entry_size'], axis=1)
            
            # Ensure we have enough data for train/test split
            if len(dataset_data) < 5:
                print(f"Warning: Dataset {dataset_name} in batch has insufficient data")
                dataset_result = {
                    'dataset_name': dataset_name,
                    'entry_size': entry_size,
                    'row_count': len(dataset_data),
                    'xgb_average': 1.0,
                    'results': {
                        "XGBoost": {"mse": 0.0, "predictions": [1.0] * 7},
                        "GBR": {"mse": 0.0, "predictions": [1.0] * 7}
                    }
                }
            else:
                # Train models on this individual dataset
                X_train, X_test, y_train, y_test = prepare_data(dataset_data)
                results = train_and_predict(X_train, X_test, y_train, y_test)
                
                # Calculate XGBoost average for this dataset
                xgb_avg = np.mean(results["XGBoost"]["predictions"])
                if np.isnan(xgb_avg) or xgb_avg <= 0:
                    xgb_avg = 1.0
                
                dataset_result = {
                    'dataset_name': dataset_name,
                    'entry_size': entry_size,
                    'row_count': len(dataset_data),
                    'xgb_average': float(xgb_avg),
                    'results': results
                }
            
            dataset_results.append(dataset_result)
        
        return {
            'batch_name': batch_info['name'],
            'batch_number': batch_info['batch_number'],
            'total_row_count': batch_info['row_count'],
            'datasets_in_batch': batch_info['datasets_in_batch'],
            'dataset_results': dataset_results  # Individual results for each dataset
        }
        
    except Exception as e:
        print(f"Error processing batch {batch_info['name']}: {str(e)}")
        # Return error structure with dummy data for each dataset
        error_dataset_results = []
        for dataset_name, entry_size in zip(batch_info['dataset_names'], batch_info['entry_sizes']):
            error_dataset_results.append({
                'dataset_name': dataset_name,
                'entry_size': entry_size,
                'row_count': 0,
                'xgb_average': 1.0,
                'results': {
                    "XGBoost": {"mse": 0.0, "predictions": [1.0] * 7},
                    "GBR": {"mse": 0.0, "predictions": [1.0] * 7}
                }
            })
        
        return {
            'batch_name': batch_info['name'],
            'batch_number': batch_info['batch_number'],
            'total_row_count': batch_info['row_count'],
            'datasets_in_batch': batch_info['datasets_in_batch'],
            'dataset_results': error_dataset_results
        }

def parse_system_info_from_output(output_text):
    """Parse system information from the C program output."""
    lines = output_text.split('\n')
    system_info = {}
    in_system_info = False
    
    for line in lines:
        line = line.strip()
        if line == "SYSTEM_INFO_START":
            in_system_info = True
        elif line == "SYSTEM_INFO_END":
            in_system_info = False
        elif in_system_info and "=" in line:
            key, value = line.split("=", 1)
            try:
                system_info[key] = int(value)
            except ValueError:
                system_info[key] = value
    
    return system_info


def run_buddy_allocation(percentages, entry_sizes, batch_number=1):
    """Run the BuddyAllocation program with the given percentages and entry sizes."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buddy_executable = os.path.join(script_dir, "BuddyAllocation")

    # Create a predictions directory if it doesn't exist
    predictions_dir = os.path.join(script_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Create the percentages file with both percentages and block sizes
    percentages_file = os.path.join(predictions_dir, f"percentages_batch_{batch_number}.txt")
    with open(percentages_file, "w") as f:
        # Write percentages first
        for percentage in percentages:
            f.write(f"{percentage:.2f}\n")
        
        # Write block sizes (entry sizes) second
        for entry_size in entry_sizes:
            f.write(f"{entry_size}\n")

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

            # Parse system information from the full output
            system_info = parse_system_info_from_output(run_result.stdout)
            
            # Extract just the results section for display
            output_lines = run_result.stdout.split("\n")
            result_section = []
            results_found = False

            for line in output_lines:
                if line.strip() == "Results:":
                    results_found = True
                    result_section.append(line)
                elif results_found:
                    result_section.append(line)

            return {
                "results_output": "\n".join(result_section),
                "system_info": system_info,
                "full_output": run_result.stdout
            }
        else:
            return {
                "results_output": "BuddyAllocation executable not found. It should be compiled during the build process.",
                "system_info": {},
                "full_output": ""
            }
    except subprocess.TimeoutExpired:
        return {
            "results_output": "BuddyAllocation execution timed out after 30 seconds.",
            "system_info": {},
            "full_output": ""
        }
    except subprocess.CalledProcessError as e:
        return {
            "results_output": f"Error running BuddyAllocation: {e.stderr}",
            "system_info": {},
            "full_output": ""
        }


def calculate_percentages_and_run_buddy_for_batch(batch_result):
    """Calculate percentages for datasets within ONE batch and run buddy allocator for that batch."""
    
    # Extract data from the datasets within this batch
    dataset_results = batch_result['dataset_results']
    xgb_averages = [dataset['xgb_average'] for dataset in dataset_results]
    entry_sizes = [dataset['entry_size'] for dataset in dataset_results]
    
    # Calculate weighted values (prediction * entry_size) for this batch
    weighted_values = [avg * size for avg, size in zip(xgb_averages, entry_sizes)]
    total_weighted_sum = sum(weighted_values)
    
    # Calculate percentages for this batch
    if total_weighted_sum > 0:
        percentages = [(weighted_value / total_weighted_sum) * 100 for weighted_value in weighted_values]
    else:
        # If all values are 0, distribute equally
        percentages = [100 / len(dataset_results)] * len(dataset_results)
    
    # Add percentage info to dataset results within this batch
    for i, dataset_result in enumerate(dataset_results):
        dataset_result['percentage'] = float(percentages[i])
        dataset_result['weighted_value'] = float(weighted_values[i])
    
    # Run buddy allocator for this specific batch
    buddy_result = run_buddy_allocation(percentages, entry_sizes, batch_result['batch_number'])
    
    return batch_result, buddy_result, percentages


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
        entry_sizes = []

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
            entry_size = dataset_obj.get("entry_size", 1)

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

            # Clean and validate the data
            try:
                # Ensure Day No. is numeric
                df["Day No."] = pd.to_numeric(df["Day No."], errors='coerce')
                
                # Ensure Number of entries is numeric
                df["Number of entries"] = pd.to_numeric(df["Number of entries"], errors='coerce')
                
                # Remove rows with NaN values
                df = df.dropna(subset=["Day No.", "Number of entries"])
                
                # Ensure we have enough data
                if len(df) < 1:
                    return (
                        jsonify(
                            {"error": f"Dataset {name} has no valid data after cleaning"}
                        ),
                        400,
                    )
                
                # Limit dataset size to prevent memory issues
                if len(df) > 1000:
                    df = df.sample(1000)

            except Exception as e:
                return (
                    jsonify(
                        {"error": f"Error processing dataset {name}: {str(e)}"}
                    ),
                    400,
                )

            datasets.append(df)
            dataset_names.append(name)
            entry_sizes.append(entry_size)

        # Create batches from the datasets
        print("Creating batches from datasets...")
        all_batches = create_batches_from_datasets(datasets, dataset_names, entry_sizes)
        
        print(f"Created {len(all_batches)} total batches")
        for batch in all_batches:
            print(f"Batch: {batch['name']} - {batch['row_count']} rows from {batch['datasets_in_batch']} datasets")

        # Process each batch (treating each dataset within the batch separately)
        print("Processing batches...")
        batch_results = []
        for batch_info in all_batches:
            print(f"Processing {batch_info['name']}...")
            result = process_batch(batch_info)
            batch_results.append(result)
            gc.collect()

        # Calculate percentages and run buddy allocator for each batch separately
        print("Calculating percentages and running buddy allocator for each batch...")
        final_batch_results = []
        all_buddy_outputs = []
        system_info_from_buddy = None  # Store system info from first buddy run
        
        for batch_result in batch_results:
            print(f"Running buddy allocator for {batch_result['batch_name']}...")
            processed_batch, buddy_result, batch_percentages = calculate_percentages_and_run_buddy_for_batch(batch_result)
            
            # Store system info from the first successful buddy run
            if not system_info_from_buddy and buddy_result.get("system_info"):
                system_info_from_buddy = buddy_result["system_info"]
            
            # Add buddy allocation info to the batch
            processed_batch['buddy_allocation_output'] = buddy_result.get("results_output", "")
            processed_batch['batch_percentages'] = batch_percentages
            processed_batch['system_info'] = buddy_result.get("system_info", {})
            
            final_batch_results.append(processed_batch)
            all_buddy_outputs.append({
                'batch_number': processed_batch['batch_number'],
                'batch_name': processed_batch['batch_name'],
                'buddy_output': buddy_result.get("results_output", ""),
                'percentages': batch_percentages,
                'system_info': buddy_result.get("system_info", {}),
                'ram_size': buddy_result.get("system_info", {}).get("TOTAL_RAM_SIZE"),
                'ram_size_mb': buddy_result.get("system_info", {}).get("TOTAL_RAM_SIZE_MB")
            })

        # Prepare final response with system info from buddy allocator
        response = {
            "datasets": dataset_names,
            "total_batches": len(final_batch_results),
            "batch_results": final_batch_results,
            "all_buddy_outputs": all_buddy_outputs
        }
        
        # Add system information from buddy allocator if available
        if system_info_from_buddy:
            response.update({
                "total_ram_size": system_info_from_buddy.get("TOTAL_RAM_SIZE"),
                "total_ram_size_mb": system_info_from_buddy.get("TOTAL_RAM_SIZE_MB"),
                "max_block_types": system_info_from_buddy.get("MAX_BLOCK_TYPES"),
                "iteration_count": system_info_from_buddy.get("ITERATION_COUNT"),
                "num_threads": system_info_from_buddy.get("NUM_THREADS"),
                "ram_size_formatted": f"{system_info_from_buddy.get('TOTAL_RAM_SIZE_MB', 0)} MB"
            })

        # Force garbage collection before returning
        gc.collect()

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# Add at the very end of your app.py file
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)