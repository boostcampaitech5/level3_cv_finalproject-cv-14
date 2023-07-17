from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mobile"
config.resume = False
config.output = None
config.embedding_size = 512
# config.output = "wf4m_arcface_r50"
config.sample_rate = 1.0
config.fp16 = True

config.batch_size = 128
config.verbose = 2000
config.dali = False

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.rec = "/opt/ml/project/data/faces_webface_112x112"
config.num_classes = 10571
# config.num_image = 4235242
config.num_image = 494149
config.num_epoch = 27
config.warmup_epoch = 3
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]


# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2