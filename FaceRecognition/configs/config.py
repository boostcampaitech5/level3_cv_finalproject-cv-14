from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.resume = False
config.output = "output_dir"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128

config.lr = 1e-4
config.num_epoch = 200
config.val_every = 5

config.optimizer = "sgd"
config.verbose = 2000
config.dali = False

config.using_wandb = True
config.wandb_entity = ""
config.wandb_project = ""
config.wandb_key = ""

# celeb data
config.rec = "data_dir"
config.num_classes = 75
config.num_image = 1499

config.warmup_epoch = 4
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

# freezing
config.freezing = False
config.freeze = "bn"