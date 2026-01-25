#!/bin/bash
# filepath: /home/karimnazarovj/gptuner/scripts/tpch_run_script.sh

# Create logs directory if it doesn't exist
mkdir -p logs/tpch_logs

# Function to run with timing - SYNCHRONOUS (wait for completion)
run_with_timing() {
    local seed=$1
    local logfile="logs/tpch_logs/tpch_10_run${seed}.log"
    
    # Debug: Check current directory and log path
    echo "Current directory: $(pwd)"
    echo "Log file path: $logfile"
    
    echo "Starting TPC-H run with seed ${seed} at $(date)" > $logfile
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    echo "Start time (Unix): $(date +%s)" >> $logfile
    
    # TPC-H doesn't need database recreation (read-only workload)
    echo "TPC-H workload - no database recreation needed" >> $logfile
    
    # Run PostgreSQL recovery script to ensure clean state
    echo "Running PostgreSQL recovery script..." >> $logfile
    scripts/recover_postgres.sh >> $logfile 2>&1
    
    echo "PostgreSQL recovery completed, starting TPC-H optimization..." >> $logfile
    
    # Run synchronously (no nohup, no &) - wait for completion
    env PYTHONPATH=src python src/run_gptuner.py postgres tpch 600 -seed=${seed} >> $logfile 2>&1
    local exit_code=$?
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    echo "End time (Unix): $(date +%s)" >> $logfile
    echo "Finished TPC-H run with seed ${seed} at $(date). Exit code: $exit_code" >> $logfile
    echo "Completed TPC-H run ${seed} with exit code $exit_code"
}

# Kill any existing processes first
pkill -f "run_gptuner.py" 2>/dev/null

echo "Starting sequential TPC-H optimization runs..."

# Run each optimization sequentially - one completes before the next starts
run_with_timing 42
run_with_timing 67
run_with_timing 83
# run_with_timing 91
# run_with_timing 55

echo "All TPC-H runs completed! Check logs in logs/tpch_logs/"