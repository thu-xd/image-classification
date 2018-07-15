from sacred import Experiment
from config import experiment_config 
from sacred.observers import MongoObserver

ex=Experiment('fine_tune',ingredients=[experiment_config])
ex.observers.append(MongoObserver.create())


