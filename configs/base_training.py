from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "prompt_image"
    cfg.dataset.folder = "./Data/train_dataset/example_prompt_image"
    cfg.dataset.manifest = "samples.json"
    cfg.dataset.image_size = 32
    cfg.dataset.batch_size = 64
    cfg.dataset.num_workers = 4
    cfg.dataset.drop_last = True

    cfg.model = ConfigDict()
    cfg.model.image_size = 32
    cfg.model.patch_size = 2
    cfg.model.dim = 256
    cfg.model.depth = 6
    cfg.model.heads = 4
    cfg.model.dropout = 0.0
    cfg.model.text_embed_dim = 512

    cfg.conditioning = ConfigDict()
    cfg.conditioning.use_prompt_id = False

    cfg.text_encoder = ConfigDict()
    cfg.text_encoder.enabled = True
    cfg.text_encoder.type = "clip"
    cfg.text_encoder.model_name = "openai/clip-vit-base-patch32"
    cfg.text_encoder.max_length = 77

    cfg.kinetic = ConfigDict()
    cfg.kinetic.lambda_const = 2.0
    cfg.kinetic.rho = 2.0
    cfg.kinetic.tau_min = 0.05
    cfg.kinetic.tau_max = 0.05
    cfg.kinetic.num_quad = 128

    cfg.optimizer_r = ConfigDict()
    cfg.optimizer_r.name = "adamw"
    cfg.optimizer_r.lr = 1e-4
    cfg.optimizer_r.betas = (0.9, 0.95)
    cfg.optimizer_r.weight_decay = 0.03
    cfg.optimizer_r.eps = 1e-8

    cfg.optimizer_s = ConfigDict()
    cfg.optimizer_s.name = "adamw"
    cfg.optimizer_s.lr = 1e-4
    cfg.optimizer_s.betas = (0.9, 0.95)
    cfg.optimizer_s.weight_decay = 0.03
    cfg.optimizer_s.eps = 1e-8

    cfg.train = ConfigDict()
    cfg.train.steps = 50000
    cfg.train.grad_clip_norm = 1.0
    cfg.train.log_every = 20
    cfg.train.save_every = 200
    cfg.train.keep_last_checkpoints = 5
    cfg.train.output_dir = "./Model/checkpoints/base_train"

    cfg.preview = ConfigDict()
    cfg.preview.enabled = True
    cfg.preview.every = 200
    cfg.preview.batch_size = 4
    cfg.preview.steps = 30
    cfg.preview.tau = 0.05
    cfg.preview.eta = 0.0

    cfg.wandb = ConfigDict()
    cfg.wandb.enabled = True
    cfg.wandb.project = "Momentum-Flow"
    cfg.wandb.entity = "MAIR_HUST"
    cfg.wandb.run_name = "base_train"
    cfg.wandb.mode = "online"
    cfg.wandb.dir = "./wandb"
    cfg.wandb.tags = ()
    cfg.wandb.notes = ""

    return cfg
