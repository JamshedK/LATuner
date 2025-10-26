# LATuner
An LLM-enhanced Database Tuning System based on Adaptive Surrogate Model

## Quick Start

### Prerequisites
- PostgreSQL 14.19 (or compatible version)
- Python 3.7.16 environment with required dependencies
- BenchBase for benchmark workloads

### Setup Instructions

1. **Configure Workload**
   - Edit `workloads.json` with the name of the benchmark you want to run
   - The workload name should match the key in your workloads.json file
   - Example workloads.json structure:
     ```json
     {
         "benchbase_tpcc_50_16": {
             "cmd": "/home/karimnazarovj/LATuner/run_benchbase.sh tpcc {0} {1} {2}"
         }
     }
     ```
   - The system uses the `run_benchbase.sh` script to execute benchmarks
   - Choose from available benchmarks (e.g., TPC-C, TPC-H, etc.)

2. **Update Database Configuration**
   - Edit `config/config.ini` with your database connection details and system specifications:
     ```ini
     [DEFAULT]
     host = localhost
     port = 5432
     database_name = your_database_name
     user = postgres
     password = your_password
     data_path = /var/lib/postgresql/14/main
     workload = your_workload_name
     objective = tps
     method = llm_end2end
     stress_test_duration = 120

     [TUNER_CONFIGS]
     cpu_cores = 4
     memory_gb = 32
     disk_gb = 250
     disk_type = SSD
     workload_type = TPC-C
     workload_size_mb = 5000
     database_type = PostgreSQL
     ```

3. **Setup BenchBase and Load Database**
   - Configure BenchBase by editing the appropriate config file in `benchbase/config/`
     - For TPC-C: Edit `benchbase/config/postgres/sample_tpcc_config.xml`
     - For TPC-H: Edit `benchbase/config/postgres/sample_tpch_config.xml`
     - Update database connection details (host, port, username, password, database name)
   - Use BenchBase to create and populate your database with the chosen benchmark:
     ```bash
     # Example for TPC-C
     cd benchbase
     java -jar benchbase-postgres.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create=true --load=true
     ```
   - Make sure the database specified in `config.ini` exists and is properly loaded

4. **Run the Tuner**
   ```bash
   python main.py
   ```

### Configuration Files

- `config/config.ini` - Main configuration file with two sections:
  - `[DEFAULT]` - Database connection and basic tuning parameters
  - `[TUNER_CONFIGS]` - Hardware specifications and workload details used by the LLM for intelligent tuning decisions
- `workloads.json` - Workload specifications and benchmark definitions
- `postgres_knobs_llm.json` - PostgreSQL tuning knobs configuration for LLM tuning
- `mysql_knobs_llm.json` - MySQL tuning knobs configuration for LLM tuning

### Hardware Configuration

The `[TUNER_CONFIGS]` section allows you to specify your system's hardware characteristics, which the LLM uses to make more informed tuning decisions:

- `cpu_cores` - Number of CPU cores available
- `memory_gb` - Amount of RAM in GB
- `disk_gb` - Disk capacity in GB
- `disk_type` - Storage type (SSD, HDD, NVMe)
- `workload_type` - Type of workload (TPC-C, TPC-H, OLTP, OLAP)
- `workload_size_mb` - Size of the workload dataset in MB
- `database_type` - Database system (PostgreSQL, MySQL)

These specifications are automatically incorporated into LLM prompts to generate hardware-appropriate tuning recommendations.

### Environment Variables

Before running, set your OpenAI API key:
```bash
export latuner_openai_api_key="your-openai-api-key-here"
```

### Output

The tuner will:
- Connect to your database using the specified configuration
- Use LLM intelligence with your hardware specifications to make informed tuning decisions
- Run the hybrid end-to-end tuning process combining LLM recommendations with Bayesian Optimization
- Log progress and results during the tuning process
- Generate performance results in the `Tuning/` directory
- Optimize database parameters based on the specified workload and objective

### Tuning Methods

The system supports different tuning methods (configured via `method` in config.ini):
- `llm_end2end` - Hybrid approach using LLM + Bayesian Optimization + Multi-Armed Bandits (recommended)
- `llm_assist` - LLM-assisted Bayesian Optimization
- `origin` - Pure Bayesian Optimization without LLM