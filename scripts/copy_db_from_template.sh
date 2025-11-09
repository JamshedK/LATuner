#!/bin/bash
# filepath: /home/karimnazarovj/gptuner/scripts/copy_db_from_template.sh

echo "Recreating tpcc_50 database from template..."

# Connect to PostgreSQL and recreate the database
# Step 1: Terminate connections
PGPASSWORD=123456 psql -h localhost -p 5432 -U postgres -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'tpcc_50' AND pid <> pg_backend_pid();
"

# Step 2: Drop the existing database
PGPASSWORD=123456 psql -h localhost -p 5432 -U postgres -c "DROP DATABASE IF EXISTS tpcc_50;"

# Step 3: Create new database from template
PGPASSWORD=123456 psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE tpcc_50 WITH TEMPLATE tpcc_50_template;"

# Step 4: Check database size
echo "Checking database size..."
PGPASSWORD=123456 psql -h localhost -p 5432 -U postgres -c "SELECT pg_size_pretty(pg_database_size('tpcc_50')) AS size;"

echo "Database tpcc_50 recreated from tpcc_50_template successfully!"
echo "You can now run the TPC-C optimization script."
