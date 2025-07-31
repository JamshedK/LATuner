from MySQLEnv import MySQLEnv
from tuner import LLMTuner

def llm_tuning_end2end():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_1_16', 'tps', 'llm_end2end', 120, '/home/karimnazarovj/LATuner/template_docker.cnf', '/home/karimnazarovj/LATuner/my_docker.cnf')
    llm_tuner = LLMTuner('/home/karimnazarovj/LATuner/mysql_knobs_llm.json', 60, dbenv, 100, None, 'tps', 10, 5)
    logger = dbenv.logger
    logger.warn("llm end2end tuning begin!!!")
    llm_tuner.tune_end2end()
    logger.warn("llm end2end tuning over!!!")

if __name__ == "__main__":

    # Run the end-to-end tuning test
    llm_tuning_end2end()