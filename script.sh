#!/bin/bash

# Activate the virtual environment
source "$HOME/optimisation2/.venv/bin/activate"

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/optimisation2"

sigma=0.12
sigma_str=$(printf "%.2f" "$sigma")

# Function to run the command for a single N value
run_command() {
    N=$1
    data_path="$HOME/test_results/cifar10/noise_$sigma_str/N_$N/certification_output.tsv"
    outfile_path="$HOME/test_results/cifar10/noise_$sigma_str/N_$N/transform_output.tsv"
    log_path="$HOME/test_results/cifar10/noise_$sigma_str/N_$N/transform_log.log"

    # Create necessary directories
    mkdir -p "$(dirname "outfile_path")"
    mkdir -p "$(dirname "$log_path")"

    command="python main_basic.py \
    --data_path \"$data_path\" \
    --outfile \"$outfile_path\" \
    --log \"$log_path\""

    echo "$command"
    eval "$command"
}

# Set the maximum number of parallel jobs
max_jobs=32

# Loop through N values and run commands in parallel
for N in $(seq 100 100 1000)
do
    # Run the command in the background
    run_command "$N" &

    # Limit the number of parallel jobs
    if (( $(jobs -p | wc -l) >= max_jobs )); then
        wait -n
    fi
done

# Wait for all background jobs to finish
wait

# Deactivate the virtual environment
deactivate