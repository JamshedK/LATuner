import psycopg2
import json
import subprocess
import time
import os
import glob
import pandas as pd
from logger import SingletonLogger
from torch.utils.tensorboard import SummaryWriter

def get_knobs(path):
    f = open(path, 'r')
    knobs_json = json.load(f)
    return knobs_json

class MyPostgresEnv:
    def __init__(self, config, path):
        self.host = config['host']
        self.port = int(config['port'])
        self.db_name = config['database_name']
        self.user = config['user']
        self.password = config['password']
        self.data_path = config['data_path']
        self.knobs = get_knobs(path)

        # environment parameters
        self.workload = config['workload']
        self.objective = config['objective']
        self.method = config['method']
        self.stress_test_duration = config['stress_test_duration']
        self.tolerance_time = 20

        self._initial()
    
    def _initial(self):
        self.timestamp = time.time()    
        self.round = 0
        # Use current directory instead of /LATuner/
        results_save_dir = f"/home/karimnazarovj/LATuner/Tuning/{self.workload}_{self.timestamp}"
        self.results_save_dir = results_save_dir
        # Use os.makedirs for robust directory creation
        os.makedirs(results_save_dir, exist_ok=True)
        self.metric_save_path = os.path.join(results_save_dir, f'results_{self.objective}.res')
        self.dbenv_log_path = os.path.join(results_save_dir, 'dbenv.log')
        self.stress_results = os.path.join(results_save_dir, 'stress_results')
        self.stress_logs = os.path.join(results_save_dir, 'stress_logs')
        # Use current directory
        self.tensorboard_logs = os.path.join("/home/karimnazarovj/LATuner/Tuning", 'tb_logs')        # Use os.makedirs with exist_ok=True
        os.makedirs(self.stress_results, exist_ok=True)
        os.makedirs(self.stress_logs, exist_ok=True)
        os.makedirs(self.tensorboard_logs, exist_ok=True)
        self.logger = SingletonLogger(self.dbenv_log_path).logger
        self.writer = SummaryWriter(log_dir=self.tensorboard_logs, flush_secs=10)
        self.perfs = {}
        self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'], self.perfs["last_best_tps"] = None, None, None, None
        self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'], self.perfs["last_best_lat"] = None, None, None, None

    def get_conn(self, max_retries=3):
        """Get PostgreSQL connection with retry logic"""
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(
                    database=self.db_name,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=int(self.port),
                    connect_timeout=10  # Add timeout
                )
                if attempt > 0:
                    self.logger.info(f"Connection successful on attempt {attempt + 1}")
                return conn
                
            except psycopg2.OperationalError as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in 2 seconds... ({attempt + 2}/{max_retries})")
                    time.sleep(2)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} attempts")
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in 2 seconds... ({attempt + 2}/{max_retries})")
                    time.sleep(2)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} attempts")
                    raise
    
    def fetch_knob(self):
        conn = self.get_conn()
        knobs = {}
        cursor = conn.cursor()
        for knob in self.knobs:
            sql = "SELECT name, setting FROM pg_settings WHERE name='{}'".format(knob)
            cursor.execute(sql)
            result = cursor.fetchall()
            for s in result:
                knobs[knob] = float(s[1])
        cursor.close()
        conn.close()
        return knobs

    def apply_knobs(self, knobs=None):
        """3-step process: Reset -> Set -> Restart"""
        if knobs is None:
            knobs = {}
            
        self.logger.info(f"Applying knobs with 3-step process: {knobs}")
        
        # Step 1: Reset all knobs to defaults
        self.logger.info("Step 1: Resetting all knobs to defaults")
        self.reset_all_knobs()
        
        # Step 2: Set new knobs
        self.logger.info("Step 2: Setting new knobs")
        self.change_knob(knobs)
        
        # Step 3: Restart PostgreSQL
        self.logger.info("Step 3: Restarting PostgreSQL")
        success = self.restart_postgres()
        
        return success

    def reset_all_knobs(self):
        """Reset all knobs to defaults using ALTER SYSTEM"""
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            conn.autocommit = True  # Enable autocommit for ALTER SYSTEM commands
            
            try:
                sql = "ALTER SYSTEM RESET ALL;"
                cursor.execute(sql)
                self.logger.info("Reset all knobs to default")
                
                # No need for conn.commit() - autocommit handles it
                cursor.execute("SELECT pg_reload_conf();")

            
                self.logger.info("All knobs reset to defaults!")
                
            except Exception as error:
                self.logger.error(f"Error resetting knobs: {error}")
            finally:
                cursor.close()
                conn.close()
                
        except Exception as error:
            self.logger.error(f"Could not connect to database for resetting knobs: {error}")


    def restart_postgres(self):
        """Restart PostgreSQL using pg_ctlcluster"""
        try:
            # Stop PostgreSQL
            self.logger.info("Stopping PostgreSQL...")
            subprocess.run(["sudo", "pg_ctlcluster", "12", "main", "stop"], 
                        capture_output=True, timeout=30)
            time.sleep(3)
            
            # Start PostgreSQL
            self.logger.info("Starting PostgreSQL...")
            subprocess.run(["sudo", "pg_ctlcluster", "12", "main", "start"], 
                        capture_output=True, timeout=30)
            
            # Wait for PostgreSQL to be ready
            self.logger.info("Waiting for PostgreSQL to be ready...")
            for attempt in range(20):
                try:
                    time.sleep(2)
                    conn = self.get_conn()
                    conn.close()
                    self.logger.info("PostgreSQL is ready!")
                    self.dbsize = self.get_db_size()
                    return True
                except Exception as e:
                    if attempt == 19:
                        self.logger.error(f"PostgreSQL failed to start: {e}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error restarting PostgreSQL: {e}")
            return False

    def change_knob(self, knobs):
        """
        Apply knob changes without SSH - PostgreSQL only
        """
        flag = True
        conn = self.get_conn()
        cursor = conn.cursor()
        conn.autocommit = True  # Enable autocommit for ALTER SYSTEM commands
        
        try:
            for knob in knobs:
                val = knobs[knob]
                
                # Convert value to appropriate type
                if self.knobs[knob]['type'] == 'integer':
                    val = int(val)
                elif self.knobs[knob]['type'] == 'real':
                    val = float(val)
                
                try:
                    # Use ALTER SYSTEM to change configuration
                    sql = "ALTER SYSTEM SET {} = %s;".format(knob)
                    cursor.execute(sql, (val,))
                    print(f"Set {knob} = {val}")
                    self.logger.info(f"Set {knob} = {val}")
                    
                except Exception as error:
                    print(f"Error setting {knob} = {val}: {error}")
                    flag = False
                        
            # Reload configuration
            cursor.execute("SELECT pg_reload_conf();")
            
            if flag:
                self.logger.info('Applied knobs successfully!')
            else:
                self.logger.error('Some knobs failed to apply')
                
        except Exception as error:
            print(f"Error applying knobs: {error}")
            flag = False
        finally:
            cursor.close()
            conn.close()
        
        return flag
    
    def get_db_size(self):
        """Get database size in MB"""
        conn = self.get_conn()
        sql = "SELECT ROUND(pg_database_size(%s) / 1024.0 / 1024.0, 2)"
        cursor = conn.cursor()
        cursor.execute(sql, (self.db_name,))
        result = cursor.fetchall()
        db_size = float(result[0][0])
        cursor.close()
        conn.close()
        return db_size
    
    def copy_db(self):
        """Recreate database from template if objective is tps"""
        if self.objective == 'tps':
            self.logger.info("Recreating database from template for TPC-C")
            script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'copy_db_from_template.sh')
            cmd = [script_path, self.db_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Failed to recreate database: {result.stderr}")
                raise Exception(f"Database recreation failed: {result.stderr}")
            self.logger.info("Database recreated successfully")
    
    def step(self, knobs=None):
        """Main step function - apply knobs and get metrics"""
        self.logger.info(f"=== Starting round {self.round} ===")
        
        # Recreate database from template if needed
        # self.copy_db()
        
        # Apply knobs
        flag = self.apply_knobs(knobs)
        if not flag:
            self.logger.error("Failed to apply knobs, skipping this round")
            return None
            
        # Get performance metrics
        metrics = self.get_metrics()
        if metrics is None:
            self.logger.error("Failed to get metrics")
            return None
            
        try:
            if self.workload.startswith("benchbase"):
                metrics["tps_std"] = self.tps_std
                metrics["lat95_std"] = self.lat_std
                metrics['knobs'] = knobs
                metrics['dbsize'] = self.dbsize
                tmp_tps = metrics["Throughput (requests/second)"]
                tmp_lat = metrics["Latency Distribution"]["95th Percentile Latency (microseconds)"]
                if not self.perfs['cur_tps']:
                    self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'], self.perfs['last_best_tps'] = tmp_tps, tmp_tps, tmp_tps, tmp_tps
                else:
                    self.perfs['cur_tps'] = tmp_tps
                    if self.perfs['best_tps'] < tmp_tps:
                        self.perfs['last_best_tps'] = self.perfs['best_tps']
                        self.perfs['best_tps'] = tmp_tps

                if not self.perfs['cur_lat']:
                    self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'], self.perfs['last_best_lat'] = tmp_lat, tmp_lat, tmp_lat, tmp_lat
                else:
                    self.perfs['cur_lat'] = tmp_lat
                    if self.perfs['best_lat'] > tmp_lat:
                        self.perfs['last_best_lat'] = self.perfs['best_lat']
                        self.perfs['best_lat'] = tmp_lat
                    
                self.writer.add_scalars(f"tps_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_tps'], 'best': self.perfs['best_tps'], 'default': self.perfs['default_tps']}, self.round)
                self.writer.add_scalars(f"lat_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_lat'], 'best': self.perfs['best_lat'], 'default': self.perfs['default_lat']}, self.round)
            else:
                pass
        except Exception as e:
            tmp_tps = -0x3f3f3f3f
            tmp_lat = 0x3f3f3f3f
            print(e)
        
        self.save_running_res(metrics)
        self.logger.info(f"save running res to {self.metric_save_path}")
        self.logger.info(f"round {self.round} over!!!")
        self.round += 1

        return tmp_tps if self.objective == "tps" else tmp_lat

    def get_workload_info(self):
        with open("./workloads.json", "r") as f:
            infos = json.load(f)
        if self.workload.startswith("benchbase"):
            # log the workload name 
            self.logger.info(f"Using workload: {self.workload}")
            infos[self.workload]["cmd"] = infos[self.workload]["cmd"].format(time.time(), self.stress_results, self.stress_logs)
            return infos[self.workload]["cmd"]
        else:
            pass

    def get_metrics(self):
        cmd = self.get_workload_info()
        self.logger.info(f"get workload stress test cmd: {cmd}")
        try:
            self.logger.info("begin workload stress test")
            p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            try:
                outs, errs = p_benchmark.communicate(timeout=int(self.stress_test_duration) + int(self.tolerance_time))
                ret_code = p_benchmark.poll()
                if ret_code == 0:
                    self.logger.info("benchmark finished!")
                else:
                    self.logger.error(f"benchmark failed with return code {ret_code}")
                    self.logger.error(f"stdout: {outs.decode()}")
                    self.logger.error(f"stderr: {errs.decode() if errs else 'No stderr output'}")
                    return None
            except Exception as e: 
                self.logger.info("Some error happened during stress test")
                self.logger.info(f"{e}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to start the benchmark process {e}")
            return None

        self.logger.info("clean extra files and get metrics file path")
        outfile_path = self.clean_and_find()
        self.logger.info("parser metrics file")
        metrics = self.parser_metrics(outfile_path)
        return metrics

    def clean_and_find(self):
        files = os.listdir(self.stress_results)
        if self.workload.startswith("benchbase"):
            info_files = [file for file in files if file.endswith("samples.csv")]
            info_file = sorted(info_files)[-1]
            df = pd.read_csv(os.path.join(self.stress_results, info_file))
            self.tps_std = df["Throughput (requests/second)"].std()
            self.lat_std = df["95th Percentile Latency (microseconds)"].std()
            for file in files:
                if not file.endswith("summary.json"):
                    os.remove(os.path.join(self.stress_results, file))

            files = [file for file in files if file.endswith("summary.json")]
            files = sorted(files)
            return os.path.join(self.stress_results, files[-1])
        else:
            pass

    def parser_metrics(self, path):
        if self.workload.startswith("benchbase"):
            with open(path, "r") as f:
                metrics = json.load(f)
        else:
            pass
        return metrics

    def save_running_res(self, metrics):
        if self.workload.startswith("benchbase"):
            save_info = json.dumps(metrics)
            with open(self.metric_save_path, 'a+') as f:
                f.write(save_info + '\n')
                f.flush()
        else:
            pass
