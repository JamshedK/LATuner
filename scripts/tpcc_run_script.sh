#!/bin/bash
# LATuner TPC-C run script

# Create logs directory if it doesn't exist
mkdir -p logs

# Kill any existing processes first
pkill -f "main.py" 2>/dev/null

echo "Starting LATuner TPC-C optimization runs..."

# Run 3 trials sequentially
for trial in 1; do
    logfile="logs/latuner_test${trial}.log"
    
    echo "==================================="
    echo "Starting Trial ${trial} at $(date)"
    echo "Log file: $logfile"
    echo "==================================="
    
    # Log start time to file
    echo "Starting LATuner Trial ${trial} at $(date)" > $logfile
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    START_UNIX=$(date +%s)
    echo "Start time (Unix): $START_UNIX" >> $logfile
    echo "" >> $logfile
    
    # Run PostgreSQL recovery script to ensure clean state
    echo "Running PostgreSQL recovery script..."
    bash scripts/recover_postgres.sh
    
    echo "Starting LATuner optimization..."
    
    # Run with nohup and wait for completion
    nohup python -u main.py >> $logfile 2>&1 &
    PID=$!
    
    echo "Trial ${trial} started with PID: $PID"
    
    # Wait for the process to complete
    wait $PID
    EXIT_CODE=$?
    
    # Log end time and duration
    END_UNIX=$(date +%s)
    DURATION=$((END_UNIX - START_UNIX))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    echo "" >> $logfile
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    echo "End time (Unix): $END_UNIX" >> $logfile
    echo "Duration: ${DURATION} seconds (${DURATION_MIN}m ${DURATION_SEC}s)" >> $logfile
    echo "Exit code: $EXIT_CODE" >> $logfile
    
    echo "Trial ${trial} completed with exit code: $EXIT_CODE at $(date)"
    echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
    echo ""
done

echo "All LATuner trials completed! Check logs in logs/"