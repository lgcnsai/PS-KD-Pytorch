import argparse
import json
import pickle
import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import HyperBand as HB

from main import main


class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_dict = {"lr": 0.2, "weight_decay": 5e-4, "start_epoch": 0, "end_epoch": 300, "batch_size": 256,
                             "experiments_dir": "models", "classifier_type": "SmallResNet", "data_path": 'data',
                             "data_type": 'cifar100', "alpha_T": 0.8, "cosine_schedule": True, "saveckp_freq": 300,
                             "workers": 40, "custom_transform": False, "use_teacher_loss": True,
                             "use_student_loss": True, "temperature": 1.0, "kill_similar_gradients": False,
                             "use_prior": False, "sim_threshold": 1.0, "dis_sim_threshold": 1.0,
                             "teacher_lr": 0.2, "teacher_weight_decay": 1e-6, "resume": None}

    def compute(self, config, budget, **kwargs):
        for key, value in self.default_dict.items():
            if key not in config:
                config[key] = value
        config["end_epoch"] = int(budget)
        args = argparse.Namespace(**config)
        loss = main(args)
        return ({
            'loss': float(-loss),  # Bohb tries to minimize
            'info': loss  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        learning_rate_student = UniformFloatHyperparameter('lr', 0.2, 0.3, log=True)
        weight_decay_student = UniformFloatHyperparameter('weight_decay', 5e-5, 5e-3, log=True)
        learning_rate_teacher = UniformFloatHyperparameter('teacher_lr', 0.2, 0.3, log=True)
        weight_decay_teacher = UniformFloatHyperparameter('teacher_weight_decay', 1e-7, 1e-4, log=True)
        temperature = UniformFloatHyperparameter('temperature', 0.5, 16, log=True)
        config_space.add_hyperparameters([learning_rate_student, weight_decay_student,
                                          learning_rate_teacher, weight_decay_teacher, temperature])
        return config_space


NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)
random_search = HB(configspace=w.get_configspace(), run_id='example1', nameserver='127.0.0.1', min_budget=33,
                   max_budget=300)
res = random_search.run(n_iterations=1)
random_search.shutdown(shutdown_workers=True)
NS.shutdown()
# get the "dict" that translates config ids to the actual configuration
id2conf = res.get_id2config_mapping()

# Here is how you get the incumbent (best configuration)
inc_id = res.get_incumbent_id()

inc_runs = res.get_runs_by_id(inc_id)
inc_run = inc_runs[0]
inc_config = id2conf[inc_id]["config"]
inc_info = inc_run.info
overall_result = {"configuration": inc_config, "info": inc_info}
with open(f"bohb_run.json", "w") as f:
    json.dump(overall_result, f)
with open(f"bohb_run.pkl", "wb") as f:
    pickle.dump(res, f)