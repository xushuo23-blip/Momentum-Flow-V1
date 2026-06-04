from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()
    cfg.checkpoint = "./Model/checkpoints/base_train/checkpoint.200.pt"
    cfg.train_config = "./configs/base_training.py"
    cfg.prompt_file = "./Data/sample_dataset/example_prompts/prompts.json"
    cfg.prompts = ()
    cfg.batch_size = 16
    cfg.steps = 100
    cfg.tau = None
    cfg.eta = 0.0
    cfg.prompt_id = None
    cfg.out = "./Sampling/outputs/samples.png"
    return cfg
