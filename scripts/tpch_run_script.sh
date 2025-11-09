#!/bin/bash

# Check if database name parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <database_name>"
    echo "Example: $0 tpch_5"
    exit 1
fi

DB_NAME=$1

# Create logs directory if it doesn't exist (in parent directory)
mkdir -p ../logs/tpch_logs
echo "Created logs directory, checking if it exists..."
ls -la ../logs/

# Function to run with timing
run_with_timing() {
    local run_id=$1
    local logfile="../logs/tpch_logs/${DB_NAME}_run${run_id}.log"
    
    echo "Start time (Unix): $(date +%s)" > $logfile
    
    echo "Current working directory: $(pwd)" >> $logfile

    # Reset PostgreSQL knobs to default
    ./recover_postgres.sh >> $logfile 2>&1
    
    # Recreate database from template
    ./copy_db_from_template.sh >> $logfile 2>&1
    
    echo "Running TPCH benchmark on database: $DB_NAME" >> $logfile
    # run main.py 
    cd .. 
    python main.py >> $logfile 2>&1
    cd scripts
    
    echo "End time (Unix): $(date +%s)" >> $logfile
    echo "Run ${run_id} completed"
}

# Kill any existing processes first
pkill -f "main.py" 2>/dev/null

echo "Starting 5 TPCH runs for: $DB_NAME"
cd scripts
# Run 5 times
for i in 1 2 3 4 5; do
    echo "Starting run $i..."
    run_with_timing $i
    # wait for a few seconds between runs
    sleep 5
done

echo "All runs completed! Check logs in ../logs/tpch_logs/"