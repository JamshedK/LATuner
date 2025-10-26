import json
import time
import os
import re

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf

from openai import OpenAI
from mab import ThompsonSamplingBandit
import requests

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


def transform_knobs2vector(knobs_detail, knobs):
    keys = list(knobs.keys())
    ys = []
    for key in keys:
        if knobs_detail[key]['type'] == 'integer':
            minv, maxv = knobs_detail[key]['min'], knobs_detail[key]['max']
            tmpv = (knobs[key] - minv) / (maxv - minv)
            ys.append(tmpv)
        elif knobs_detail[key]['type'] == 'enum':
            enum_vs = knobs_detail[key]['enum_values']
            tmpv = enum_vs.index(knobs[key]) / (len(enum_vs) - 1)
            ys.append(tmpv)
        else:
            pass
    return ys
def transform_vector2knobs(knobs_detail, vector):
    keys = list(knobs_detail.keys())
    knobs = {}
    for i in range(len(keys)):
        if knobs_detail[keys[i]]['type'] == 'integer':
            minv, maxv = knobs_detail[keys[i]]['min'], knobs_detail[keys[i]]['max']
            tmpv = (maxv - minv) * float(vector[i]) + minv
            knobs[keys[i]] = int(tmpv)
        elif knobs_detail[keys[i]]['type'] == 'enum':
            enum_vs = knobs_detail[keys[i]]['enum_values']
            tmpv = vector[i] * (len(enum_vs) - 1)
            knobs[keys[i]] = enum_vs[int(tmpv)]
        else:
            pass
    return knobs
def transform_knobs2cnf(knobs_detail, knobs):
    keys = list(knobs.keys())
    for key in keys:
        if knobs_detail[key]['type'] == 'integer':
            knobs[key] = int(knobs[key])
        else:
            pass
    return knobs

def proxy_chat(system_content, prompt):
    url = "https://api.openai-hk.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": os.getenv("latuner_openai_api_key")
    }
    data = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
    result = response.content.decode("utf-8")
    return json.loads(result)

class LLMTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None, objective='lat', warm_start_times=10, prune_nums=5, tuner_config=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "LLM"
        self.objective = objective
        self.warm_start_times = warm_start_times
        self.prune_nums = prune_nums
        self.proxy = False
        api_key = os.getenv("latuner_openai_api_key")
        self.client = proxy_chat if self.proxy else OpenAI(api_key=api_key)
        
        # Store tuner config for dynamic access
        self.tuner_config = tuner_config or {}
        
        # Helper method to get config values with defaults
        self.get_config = lambda key, default: self.tuner_config.get(key, default)
        self.system_content = '''You will be helping me with the knob tuning task for {0} database. '''
        self.user_content_prune = '''The specific information for the knobs is: {0}. The specific information for the machine on which the {1} works is: {2} cores {3} RAM and {4} disk. The specific information for the workload is: {5} size {6}. The goal of the current tuning task is to optimize {7}, please give the {8} knobs that have the greatest impact on the performance of the database. You should give these top-{8} knobs by json style.  The given knobs must be included in the previously given knobs. Just give the json without any other extra output.'''
        self.user_content_ws_samples = '''The specific information for the knobs is: {0}. The specific information for the machine on which the {1} works is: {2} cores {3} RAM and {4} disk. The specific information for the workload is: {5} size {6}. The goal of the current tuning task is to optimize {7}, please suggest {8} diverse yet effective configurations to initiate a Bayesian Optimization process for knobs tuning. You mustn't include “None” in the configurations. Your response should include a list of dictionaries, where each dictionary describes one recommended configuration.Just give the dictionaries without any other extra output.'''
        self.bandit = ThompsonSamplingBandit(num_arms=2)

    def _check_knobs_valid(self, knobs_set):
        if len(knobs_set) != 5:
            return False 
        keys = list(self.knobs_detail.keys())
        for knobs in knobs_set:
            tmp_keys = list(knobs.keys())
            for key in tmp_keys:
                if key not in keys:
                    self.logger.info(f"key: {key} not in knob space")
                    return False
            for key in tmp_keys:
                if self.knobs_detail[key]['type'] == 'integer':
                    v = knobs[key]
                    minv, maxv = self.knobs_detail[key]['min'], self.knobs_detail[key]['max']
                    if v < minv or v > maxv:
                        self.logger.info(f"knobs range error, {key} : {v}")
                        return False
                else:
                    v = knobs[key]
                    vs = self.knobs_detail[key]['enum_values']
                    if v not in vs:
                        self.logger.info(f"knobs range error, {key} : {v}")
                        return False
        return True
    
    def _gen_candidates_llm(self, nums, target):
        system_content = self.system_content.format(self.get_config('database_type', 'PostgreSQL'))
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        prompt = '''The following examples demonstrate {0} database running on a machine with {1} cores, {2} of memory, and a {3} {4} disk, under a {5} {6} workload. These examples involve adjusting various knobs configurations to observe changes in {7} metrics:\n'''.format(
            self.get_config('database_type', 'PostgreSQL'),
            self.get_config('cpu_cores', 4),
            f"{self.get_config('memory_gb', 32)}GB",
            f"{self.get_config('disk_gb', 50)}GB",
            self.get_config('disk_type', 'SSD'),
            f"{self.get_config('workload_size_mb', 100)}MB",
            self.get_config('workload_type', 'TPC-C'),
            obj_str
        )
        self.logger.info(f"Get candidates LLM start")
        self.logger.info(f"Get candidates LLM prompt: {prompt}")
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['Average Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            prompt += "Knob configuration: " + json.dumps(knobs) + "\n"
            prompt += "Performance: " + str(int(metric)) + "\n"
        prompt +=  f"The database knob space is: {json.dumps(self.knobs_detail)}." + "\n"
        prompt += f"Please recommend {nums} configurations that will result in a database {obj_str} of {int(target)}. Each knob must contained within the knob space, Your response must only contain the predicted configurations, in the format ## Knob configuration: ##."
        count = 10
        while count > 0:
            try:
                if self.proxy:
                    completion = self.client(system_content, prompt)
                else:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt}
                        ]
                    )
                strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
                pattern = re.compile(r'\{.*?\}')
                matches = pattern.findall(strs)
                samples = []
                for match in matches:
                    samples.append(eval(match))
                if self._check_knobs_valid(samples):
                    self.logger.info(f"time {10 - count}, gpt return sucess")
                    break
            except Exception as e:
                print(e)
                pass
            time.sleep(15)
            self.logger.info(f"time {10 - count}, gpt return error")
            count -= 1

        if count == 0:
            self.logger.error(f"gpt return fail")
            return
        return samples

    def _prediction_llm(self, knobs_set):
        system_content = self.system_content.format(self.get_config('database_type', 'PostgreSQL'))
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        prompt = '''The following examples demonstrate {0} database running on a machine with {1} cores, {2} of memory, and a {3} {4} disk, under a {5} {6} workload. These examples involve adjusting various knobs configurations to observe changes in {7} metrics:\n'''.format(
            self.get_config('database_type', 'PostgreSQL'),
            self.get_config('cpu_cores', 4),
            f"{self.get_config('memory_gb', 32)}GB",
            f"{self.get_config('disk_gb', 50)}GB",
            self.get_config('disk_type', 'SSD'),
            f"{self.get_config('workload_size_mb', 100)}MB",
            self.get_config('workload_type', 'TPC-C'),
            obj_str
        )
        self.logger.info(f"Prediction LLM start")
        self.logger.info(f"Predictoin LLM prompt: {prompt}")
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['Average Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            prompt += "Knob configuration: " + json.dumps(knobs) + "\n"
            prompt += "Performance: " + str(int(metric)) + "\n"
        prompt +=  f"The allowable ranges for knobs are: {json.dumps(self.knobs_detail)}. "
        prompt += "Please combine the above information to determine which of the following configurations is a high potential configuration: \n"
        for knobs in knobs_set:
            prompt += json.dumps(knobs) + "\n"
        prompt += "Your response should only contain one of the above configurations."
        count = 10
        while True:
            try:
                if self.proxy:
                    completion = self.client(system_content, prompt)
                else:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt}
                        ]
                    )
                strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
                pattern = re.compile(r'\{.*?\}')
                matches = pattern.findall(strs)
                samples = []
                for match in matches:
                    samples.append(eval(match))
                if len(samples) > 0:
                    break
            except Exception as e:
                print(e)
            count -= 1

        return samples[0]

    def _knob_prune(self, nums):
        knobs_str = json.dumps(self.knobs_detail)
        system_content = self.system_content.format(self.get_config('database_type', 'PostgreSQL'))
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        user_content = self.user_content_prune.format(
            knobs_str, 
            self.get_config('database_type', 'PostgreSQL'),
            self.get_config('cpu_cores', 4),
            f"{self.get_config('memory_gb', 32)}GB",
            f"{self.get_config('disk_gb', 50)}GB {self.get_config('disk_type', 'SSD')}",
            f"{self.get_config('workload_size_mb', 100)}MB",
            self.get_config('workload_type', 'TPC-C'),
            obj_str, 
            nums
        )
        self.logger.info(f"Knob prune start")
        self.logger.info(f"knob prune prompt: {user_content}")
        if self.proxy:
            completion = self.client(system_content, user_content)
        else:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
        strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
        keys = []
        for key in list(self.knobs_detail.keys()):
            if key in strs:
                keys.append(key)
        print(keys)
        assert(len(keys) == nums)
        KNOB_DETAILS = {}
        for key in keys:
            KNOB_DETAILS[key] = self.knobs_detail[key]
        self.knobs_detail = KNOB_DETAILS
        self.knob_nums = nums
    
    def _get_warm_start_samples(self, nums):
        knobs_str = json.dumps(self.knobs_detail)
        system_content = self.system_content.format(self.get_config('database_type', 'PostgreSQL'))
        obj_str = 'throughput' if self.objective == 'tps' else 'latency'
        user_content = self.user_content_ws_samples.format(
            knobs_str, 
            self.get_config('database_type', 'PostgreSQL'),
            self.get_config('cpu_cores', 4),
            f"{self.get_config('memory_gb', 32)}GB",
            f"{self.get_config('disk_gb', 50)}GB {self.get_config('disk_type', 'SSD')}",
            f"{self.get_config('workload_size_mb', 100)}MB",
            self.get_config('workload_type', 'TPC-C'),
            obj_str, 
            nums
        )
        self.logger.info(f"Warm start sample generation start")
        self.logger.info(f"warm start prompt: {user_content}")
        if self.proxy:
            completion = self.client(system_content, user_content)
        else:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
        strs = completion["choices"][0]["message"]["content"] if self.proxy else completion.choices[0].message.content
        print(strs)
        pattern = re.compile(r'\{.*?\}')
        matches = pattern.findall(strs)
        samples = []
        for match in matches:
            samples.append(eval(match))
        knobs = samples
        return knobs
    
    def _get_next_point_origin(self):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['Average Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        target = float(train_Y.min()) if self.objective == "lat" else float(train_Y.max())
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        #EI = ExpectedImprovement(gp, best_f=target, maximize=False if self.objective == 'lat' else True)
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=10, raw_samples=2000
            )
        knobs = transform_vector2knobs(self.knobs_detail, candidate[0])
        
        return knobs
    
    def _get_next_point_llm_assist(self, candidate_nums=5):
        data_file = self.dbenv.metric_save_path
        f = open(data_file, 'r')
        lines = f.readlines()
        f.close()
        train_X, train_Y = [], []
        for line in lines[1:]:
            line = json.loads(line)
            knobs = line['knobs']
            metric = line['Latency Distribution']['Average Latency (microseconds)'] if self.objective == 'lat' \
                     else line['Throughput (requests/second)']
            train_X.append(transform_knobs2vector(self.knobs_detail, knobs))
            train_Y.append([metric])
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        target = float(train_Y.min()) if self.objective == "lat" else float(train_Y.max())
        print(target)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        train_X = torch.tensor(train_X, dtype=torch.float64)
        train_Y = torch.tensor(train_Y, dtype=torch.float64)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        EI = ExpectedImprovement(gp, best_f=target, maximize=False if self.objective == 'lat' else True)
        bounds = torch.stack([torch.zeros(self.knob_nums), torch.ones(self.knob_nums)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidates_default, acq_values_default = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=candidate_nums, raw_samples=2000, return_best_only=False
            )
        llm_initial_samples = self._gen_candidates_llm(candidate_nums, target)
        samples = []
        for sample in llm_initial_samples:
            samples.append(transform_knobs2vector(self.knobs_detail, sample))
        if samples:
            samples = torch.tensor(samples, dtype=torch.float64)
            samples = samples.reshape(candidate_nums, 1, self.knob_nums)
            with gpytorch.settings.cholesky_jitter(1e-1):
                candidates_llm, acq_values_llm = optimize_acqf(
                    EI, bounds=bounds, q=1, num_restarts=candidate_nums, batch_initial_conditions=samples, return_best_only=False
                )
            candidates = torch.concat([candidates_default, candidates_llm], dim=0)
            acq_values = torch.concat([acq_values_default, acq_values_llm], dim=0)
        else:
            candidates = candidates_default
            acq_values = acq_values_default

        idx = int(acq_values.argmax())
        knobs_set = []
        size = candidates.shape[0]
        for i in range(size):
            candidate = candidates[i][0]
            tmp_knobs = transform_vector2knobs(self.knobs_detail, candidate)
            knobs_set.append(tmp_knobs)

        knobs = transform_vector2knobs(self.knobs_detail, candidates[idx][0])
        return knobs, knobs_set
    
    def _get_next_point_hybrid(self):
        knobs_default, knobs_set = self._get_next_point_llm_assist()
        knobs_llm = self._prediction_llm(knobs_set)
        return knobs_default, knobs_llm

    def _get_reward(self):
        if self.objective == 'lat':
            perf_first = self.dbenv.perfs['last_best_lat'] 
            perf_last = self.dbenv.perfs['cur_lat']
            if perf_first is None or perf_last is None:  # Handle None values
                return 0
            if perf_first - perf_last > 0:
                return 1
            else:
                return 0
        else:
            perf_first = self.dbenv.perfs['last_best_tps']  
            perf_last = self.dbenv.perfs['cur_tps']
            if perf_first is None or perf_last is None:  # Handle None values
                return 0
            if perf_last - perf_first > 0:
                return 1
            else:
                return 0
        

    def tune(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs = self._get_next_point_origin()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

    def tune_llm_assist(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs, _ = self._get_next_point_llm_assist()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            self.dbenv.step(knobs)

    def tune_end2end(self):
        self._knob_prune(self.prune_nums)
        knobs_set = self._get_warm_start_samples(self.warm_start_times)
        self.dbenv.step(None)
        self.logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self.dbenv.step(knobs)
        self.logger.info("warm start over!!!")
        total_reward = 0
        for _ in range(self.bugets - self.warm_start_times):
            now = time.time()
            knobs_default, knobs_llm = self._get_next_point_hybrid()
            self.logger.info(f"recommend next knobs spent {time.time() - now}s")
            chosen_arm = self.bandit.choose_arm()
            if chosen_arm == 0:
                self.logger.info(f"choose arm {chosen_arm}: default knobs!")
                self.dbenv.step(knobs_default)
            else:
                self.logger.info(f"choose arm {chosen_arm}: llm knobs!")
                self.dbenv.step(knobs_llm)

            reward = self._get_reward()
            self.logger.info(f"get reward: {reward}")
            total_reward += reward
            self.bandit.update_arm(chosen_arm, reward)
        self.logger.info(f"total reward: {total_reward}")
