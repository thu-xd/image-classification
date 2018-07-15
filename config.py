from sacred import Ingredient

experiment_config=Ingredient('config')

@experiment_config.config
def cfg():
    data_fetch_config={
        "data_dir":"./images/dset1",
        "batch_size":128,
        "train_valid_split_ratio":0.9}

    AlexNet_config={
        "num_class":65,
        "stop_gradient_layer":"fc7",
        "dropout_prob":0.5,
        "stddev":1e-4,
        "weight_decay":0.0,
        "NUM_EPOCHES_PER_DECAY":10,
        "INITIAL_LEARNING_RATE":0.005,
        "LEARNING_RATE_DECAY_FACTOR":0.9,
        "MAX_EPOCHES":1000,
        "VALID_STEPS":100,
        "batch_size":data_fetch_config['batch_size']
    }
