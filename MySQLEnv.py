import json
import subprocess
import time
import mysql.connector
import os

from shutil import copyfile
from logger import SingletonLogger
import pandas as pd
from torch.utils.tensorboard import SummaryWriter 

class MySQLEnv():
    def __init__(self, host, user, passwd, dbname, workload, objective, method, stress_test_duration, template_cnf_path, real_cnf_path):
        self.host =  host # localhost
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.workload = workload
        self.objective = objective # tpc or latency
        self.method = method  # tuning method 
        self.stress_test_duration = stress_test_duration # 60 seconds
        
        # Hardcoded Docker-compatible paths instead of /home/root3/
        self.template_cnf_path = "./template_docker.cnf"  # Docker-compatible template
        self.real_cnf_path = "./my_docker.cnf"           # Current config file
        self.container_name = "mysql57-latuner"          # ADDED: Docker container name


        self.tolerance_time = 20 #seconds
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

    def _start_mysqld(self):
        """Start MySQL container with the given config file mounted."""
        # Remove existing container if running
        subprocess.run(["sudo", "docker", "rm", "-f", self.container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Start container with updated config
        cmd = [
            "sudo", "docker", "run", "--name", self.container_name,
            "-e", "MYSQL_ROOT_PASSWORD=",  # No password
            "-e", "MYSQL_ALLOW_EMPTY_PASSWORD=yes",
            "-e", f"MYSQL_DATABASE={self.dbname}",
            "-p", "3306:3306",
            "-v", f"{os.path.abspath(self.real_cnf_path)}:/etc/mysql/my.cnf",  # mount new config
            "-v", "/home/karimnazarovj/mysql-data:/var/lib/mysql",
            "-d", "mysql:5.7.40"
        ]
        subprocess.run(cmd)

        # Wait for DB to be ready
        self.logger.info("Waiting for Dockerized MySQL to be ready...")
        count = 0
        while True:
            try:
                conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
                if conn.is_connected():
                    conn.close()
                    self.logger.info("Connected to Dockerized MySQL")
                    self.dbsize = self.get_db_size()
                    self.logger.info(f"{self.workload} DB size is {self.dbsize} MB")
                    return True
            except Exception as e:
                self.logger.warn(f"Retry connect: {e}")
                time.sleep(1)
                count += 1
                if count > 60:
                    self.logger.error("MySQL container did not become ready")
                    return False
    
    def _kill_mysqld(self):
        """Stop and remove Docker container."""
        self.logger.info("Stopping MySQL container...")
        subprocess.run(["sudo", "docker", "stop", self.container_name])
        subprocess.run(["sudo", "docker", "rm", self.container_name])
        self.logger.info("MySQL container stopped and removed")

    
    def get_db_size(self):
        db_conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(self.dbname)
        cmd = db_conn.cursor()
        cmd.execute(sql)
        res = cmd.fetchall()
        db_size = float(res[0][0][:-2])
        db_conn.close()
        return db_size
    
    def replace_mycnf(self, knobs=None):
        if knobs == None:
            copyfile(self.template_cnf_path, self.real_cnf_path)
            return
        f = open(self.template_cnf_path)
        contents = f.readlines()
        f.close()
        for key in knobs.keys():
            contents.append(f"{key}={knobs[key]}")
        strs = '\n'.join(contents)
        with open(self.real_cnf_path, 'w') as f:
            f.write(strs)
            f.flush()
        self.logger.info("replace mysql cnf file")

    def apply_knobs(self, knobs=None):
        self._kill_mysqld()
        self.replace_mycnf(knobs)
        time.sleep(10)
        success = self._start_mysqld()
        return success
    
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
    
    def parser_metrics(self, path):
        if self.workload.startswith("benchbase"):
            with open(path, "r") as f:
                metrics = json.load(f)
        else:
            pass
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


    def get_metrics(self):
        cmd = self.get_workload_info()
        self.logger.info(f"get workload stress test cmd: {cmd}")
        try:
            self.logger.info("begin workload stress test")
            p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            try:
                outs, errs = p_benchmark.communicate(timeout=self.stress_test_duration + self.tolerance_time)
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

    def step(self, knobs=None):
        self.logger.info(f"round {self.round} begin!!!")
        self.logger.info(f"ready to apply new knobs: {knobs}")
        flag = self.apply_knobs(knobs)
        self.logger.info("apply new knobs success")
        metrics = self.get_metrics()
        if metrics == None:
            self.logger.error("this round stress test fail")
            self.logger.info("round over!!!")
            return
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

    def save_running_res(self, metrics):
        if self.workload.startswith("benchbase"):
            save_info = json.dumps(metrics)
            with open(self.metric_save_path, 'a+') as f:
                f.write(save_info + '\n')
                f.flush()
        else:
            pass
