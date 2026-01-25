from MySQLEnv import MySQLEnv
from MyPostgresEnv import MyPostgresEnv
from tuner import LLMTuner
from config import parse_config

def llm_tuning_end2end():
    # load config_ini file
    config = parse_config.parse_args('/home/karimnazarovj/LATuner/config/config.ini')
    default_config = config['DEFAULT']
    tuner_config = config['TUNER_CONFIGS']  # Extract tuner-specific configs
    print(default_config)
    dbenv = MyPostgresEnv(config = default_config, path = '/home/karimnazarovj/LATuner/postgres_knobs_llm.json',)
    llm_tuner = LLMTuner('/home/karimnazarovj/LATuner/postgres_knobs_llm.json', 44, dbenv, 100, None, default_config['objective'], 10, 5, tuner_config=tuner_config)
    logger = dbenv.logger
    logger.warn("llm end2end tuning begin!!!")
    llm_tuner.tune_end2end()
    logger.warn("llm end2end tuning over!!!")

if __name__ == "__main__":

    # Run the end-to-end tuning test
    llm_tuning_end2end()