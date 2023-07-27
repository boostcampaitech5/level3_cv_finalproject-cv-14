import argparse
import logging
import os
from datetime import datetime
import wandb
import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from Crawling_Dataset import Crawling_Nomal_Dataset
from torch.utils.data import Dataset, DataLoader
# from Validation_statistic import validation
import torch.utils.data as data
from torchvision import transforms
from facenet_pytorch import MTCNN, fixed_image_standardization, InceptionResnetV1
from torchvision.transforms import Resize
from sklearn.metrics import confusion_matrix
from Embedding import Embedding_vector, Embeddings_Manager
from Label_DataFrame import Label_DataFrame

def calculate_mean_std(df) :
    p_mean = round(df[df.decision == "Yes"].distance.mean(), 4)
    p_std = round(df[df.decision == "Yes"].distance.std(), 4)
    n_mean = round(df[df.decision == "No"].distance.mean(), 4)
    n_std = round(df[df.decision == "No"].distance.std(), 4)
    print(p_mean, p_std)
    print(n_mean, n_std)
    return p_mean, p_std, n_mean, n_std

def get_threshold(p_mean, p_std, sigma=1) :
    threshold = round(p_mean + sigma * p_std, 4)
    return threshold

def get_statistic(df, printing=False) :
    cm = confusion_matrix(df.decision.values, df.prediction.values)
    # print(cm)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn)/(tn + fp +  fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    if printing == True:
        print('acc    : ', accuracy)
        print('recall : ', recall)
        print('f1     : ', f1)
        print('precision : ', precision)
    return accuracy, recall, f1, precision

def fine_tuning_threshold(model_df : Label_DataFrame,df, sigma=1) :
    p_mean, p_std, n_mean, n_std = calculate_mean_std(df)
    start = p_mean
    end = n_mean
    ths = np.arange(start, end, 0.001)
    accuracy = 0
    threshold = start
    for t in ths :
        prediction_df = model_df.get_prediction_df(threshold=t)
        acc, recall, f1, precision = get_statistic(prediction_df, printing=False)
        if accuracy < acc :
            accuracy = acc
            threshold = t
    return threshold

def validation(model, vali_data_loader, test_path) :

    model_vector= Embedding_vector(model=model)
    model_vector_imform = Embeddings_Manager(file_path=test_path, embedding_vector=model_vector, dataloader=vali_data_loader)
    model_identities = model_vector_imform.get_label_per_path_dict()
    model_path2embedding = model_vector_imform.get_path_embedding_dict()

    model_df = Label_DataFrame(identities=model_identities)
    positive_df = model_df.get_positive_df()
    negative_df = model_df.get_negative_df()
    facenet_label_df = model_df.concate()
    model_inference_df = model_df.get_inference_df(model_path2embedding)

    p_mean, p_std, n_mean, n_std = calculate_mean_std(model_inference_df)
    threshold = fine_tuning_threshold(model_df,model_inference_df, sigma=1)
    facenet_prediction_df = model_df.get_prediction_df(threshold=threshold)

    accuracy, recall, f1, precision = get_statistic(facenet_prediction_df, printing=True)

    return accuracy, recall, f1, precision

transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        Resize((160, 160)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        fixed_image_standardization
    ])

test_path = '/opt/ml/project/dataset/Celeb/test/'
test_dataset = Crawling_Nomal_Dataset(test_path, transforms=transform)
test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    wandb_name = f'{cfg.optimizer} {cfg.lr}'
    wandb.init(
        entity=cfg.wandb_entity,
               project=cfg.wandb_project,
               name = wandb_name
               )
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    wandb.config = {
        "lr" : cfg.lr,
        "optimizer" : cfg.optimizer,
        "epoch" : cfg.num_epoch,
        "batch_size" : cfg.batch_size
    }
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    # backbone = get_model(
    #     cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=cfg.num_classes).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    
    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # callback_verification = CallBackVerification(
    #     val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
    #     summary_writer=summary_writer, wandb_logger = wandb_logger
    # )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    best_acc = 0
    for epoch in range(cfg.num_epoch):
        if cfg.freezing:
            for name, param in backbone.named_parameters():
                if cfg.freeze in name.split("."):
                    param.requires_grad = False
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                # wandb.log({
                #     'Loss/Step Loss': loss.item(),
                #     'Loss/Train Loss': loss_am.avg,
                #     'Process/Step': global_step,
                #     'Process/Epoch': epoch
                # }, step=global_step)
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
        if epoch % cfg.val_every == 0:
            accuracy, recall, f1, precision = validation(model=backbone, vali_data_loader=test_data_loader, test_path=test_path)
            wandb.log({
                    'acc': accuracy,
                    'f1': f1,
                    'recall': recall,
                    'precision': precision
                },step=epoch)
        
        
        if accuracy > best_acc:
            best_acc = accuracy
            print(f"new best acc at epoch{epoch}, {accuracy}")
            path_module = os.path.join(cfg.output, f"{wandb_name}_best.pt")
            torch.save(backbone.module.state_dict(), path_module)
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))


                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
    # input = torch.Tensor(2, 3, 250, 250)
    # backbone = get_model(
    #     'r50', dropout=0.0, fp16=True, num_features=512).cuda()
    # out = backbone(input)
    # print(out.shape)
