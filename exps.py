import json
from lhs import LHSGenerator
import subprocess
import time
import mysql.connector
import os
import re
import numpy as np
from shutil import copyfile
from logger import SingletonLogger
import queue
import pandas as pd
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qKnowledgeGradient
from botorch.optim import optimize_acqf
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import LatinHypercubeInitialDesign
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from pathlib import Path
from openai import OpenAI
from mab import ThompsonSamplingBandit
import requests
from MySQLEnv import MySQLEnv


class Tuner():
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        self.knobs_config_path = knobs_config_path
        self.knob_nums = knob_nums
        self.knob_idxs = knob_idxs
        self.initialize_knobs()
        self.dbenv = dbenv
        self.bugets = bugets
        self.logger = None if not self.dbenv else self.dbenv.logger
    def initialize_knobs(self):
        f = open(self.knobs_config_path)
        knob_tmp = json.load(f)
        KNOB_DETAILS = {}
        if not self.knob_idxs:
            i = 0
            while i < self.knob_nums:
                key = list(knob_tmp.keys())[i]
                KNOB_DETAILS[key] = knob_tmp[key]
                i = i + 1
        else:
            if type(self.knob_idxs[0]) == int:
                for idx in self.knob_idxs:
                    key = list(knob_tmp.keys())[idx]
                    KNOB_DETAILS[key] = knob_tmp[key]
            else:
                for key in self.knob_idxs:
                    KNOB_DETAILS[key] = knob_tmp[key]
        f.close()
        self.knobs_detail = KNOB_DETAILS

class LHSTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "LHS"
    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_gen = LHSGenerator(lhs_num, self.knobs_detail)
        lhs_configs = lhs_gen.generate_results()
        return lhs_configs
    def tune(self):
        self.dbenv.step(None)
        knobs_set = self.lhs(self.bugets)
        for knobs in knobs_set:
            self.dbenv.step(knobs)
            
class GridTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "Grid"

    def _grid_search(self, params_list, results, current_params=None):
        if current_params is None:
            current_params = []
        if not params_list:
            return current_params
        current_dimension = params_list[0]
        for value in current_dimension:
            result = self._grid_search(params_list[1:], results, current_params + [value])
            if result:
                results.append(result)
    
    def sampling(self, interval):
        knobs_list = []
        for knob_name in self.knobs_detail.keys():
            type = self.knobs_detail[knob_name]["type"]
            if type == "integer":
                minv = self.knobs_detail[knob_name]["min"]
                maxv = self.knobs_detail[knob_name]["max"]
                knobs_list.append(list(np.linspace(minv, maxv, interval, dtype=np.int32)))
            else:
                knobs_list.append(self.knobs_detail[knob_name]["enum_values"])
        results = []
        self._grid_search(knobs_list, results)
        return results
    
    def tune(self, interval=10):
        self.dbenv.step(None)
        knobs_set = self.sampling(interval)
        keys = list(self.knobs_detail.keys())
        for rd, ss in enumerate(knobs_set):
            self.logger.info(f"tuning round {rd + 1} begin!!")
            knobs = {}
            for i in range(len(keys)):
                if isinstance(ss[i], np.integer):
                    knobs[keys[i]] = int(ss[i])
                else:
                    knobs[keys[i]] = ss[i]
            self.dbenv.step(knobs)
            self.logger.info(f"tuning round {rd + 1} over!!")

class GPTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=20):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "GP"
        self.objective = objective
        self.warm_start_times = warm_start_times
    
    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_gen = LHSGenerator(lhs_num, self.knobs_detail)
        lhs_configs = lhs_gen.generate_results()
        return lhs_configs

    def _get_next_point(self):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['95th Percentile Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        #EI = ExpectedImprovement(gp, best_f=train_Y.max())
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        knobs = transform_vector2knobs(self.knobs_detail, candidate[0])
        
        return knobs
    def tune(self):
        self.dbenv.step(None)
        knobs_set = self.lhs(self.warm_start_times)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs = self._get_next_point()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

class SMACTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=20):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "SMAC"
        self.objective = objective
        self.warm_start_times = warm_start_times
    
    def _train(self, config, seed=0):
        knobs = dict(config)
        knobs = transform_knobs2cnf(self.knobs_detail, knobs)
        metric = self.dbenv.step(knobs)
        if self.objective == 'lat':
            return metric
        else:
            return -metric
    def tune(self):
        self.dbenv.step(None)
        keys = list(self.knobs_detail.keys())
        knobs = []
        for key in keys:
            if self.knobs_detail[key]["type"] == "integer":
                knobs.append(Float(key, (self.knobs_detail[key]["min"], self.knobs_detail[key]["max"]), default=self.knobs_detail[key]["default"]))
            elif self.knobs_detail[key]["type"] == "enum":
                knobs.append(Categorical(key, self.knobs_detail[key]["enum_values"], default=self.knobs_detail[key]["default"]))
            else:
                pass
        configspace = ConfigurationSpace("smac_tuning", seed=0)
        configspace.add_hyperparameters(knobs)
        scenario = Scenario(configspace, n_trials=self.bugets, output_directory=Path(self.dbenv.results_save_dir))
        smac = HyperparameterOptimizationFacade(scenario, self._train, initial_design=LatinHypercubeInitialDesign(scenario, self.warm_start_times))
        incumbent = smac.optimize()
        return incumbent
    

def grid_tuning_task(knobs_idxs=None):
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'grid', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    if not knobs_idxs:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10)
    else:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10, knobs_idxs)
    logger = dbenv.logger
    logger.warn("grid tuning begin!!!")
    grid_tuner.tune()
    logger.warn("grid tuning over!!!")

def lhs_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'lhs', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    lhs_tuner = LHSTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 1000)
    logger = dbenv.logger
    logger.warn("lhs tuning begin!!!")
    lhs_tuner.tune()
    logger.warn("lhs tuning over!!!")

def gp_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'gp', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    lhs_tuner = GPTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 100, None, 'tps', 10)
    logger = dbenv.logger
    logger.warn("gp tuning begin!!!")
    lhs_tuner.tune()
    logger.warn("gp tuning over!!!")

def smac_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'smac', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    smac_tuner = SMACTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 100, None, 'tps', 10)
    logger = dbenv.logger
    logger.warn("smac tuning begin!!!")
    smac_tuner.tune()
    logger.warn("smac tuning over!!!")

def llm_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 60, dbenv, 100, None, 'tps', 10, 5)
    logger = dbenv.logger
    logger.warn("llm tuning begin!!!")
    llm_tuner.tune()
    logger.warn("llm tuning over!!!")

def llm_assist_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm_assist', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 5, dbenv, 37, ['innodb_buffer_pool_size','innodb_write_io_threads','innodb_flush_log_at_timeout','innodb_read_io_threads','innodb_io_capacity_max'], 'tps', 0, 5)
    logger = dbenv.logger
    logger.warn("llm assist tuning begin!!!")
    llm_tuner.tune_llm_assist()
    logger.warn("llm assist tuning over!!!")

def llm_tuning_end2end():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'tps', 'llm_end2end', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    llm_tuner = LLMTuner('/home/root3/Tuning/mysql_knobs_llm.json', 60, dbenv, 100, None, 'tps', 10, 5)
    logger = dbenv.logger
    logger.warn("llm end2end tuning begin!!!")
    llm_tuner.tune_end2end()
    logger.warn("llm end2end tuning over!!!")

class TaskQueue():
    def __init__(self, nums=-1):
        self.queue = queue.Queue(nums)

    def _execute_task(self, task):
        task_func, task_args = task
        task_func(*task_args)
    
    def add(self, task):
        self.queue.put(task)
    
    def run(self):
        while not self.queue.empty():
            task = self.queue.get()
            self._execute_task(task)
    

if __name__ == '__main__':
    task_queue = TaskQueue()
    task_queue.add((llm_tuning_task, ()))
    task_queue.run()