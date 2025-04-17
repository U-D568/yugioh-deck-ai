import gc
import datetime
import logging
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ultralytics import YOLO

from utils import common, tal, losses, logger, gradNorm
from data.preprocess.tf import EmbeddingPreprocessor
from data.preprocess.torch import detector_preprocessing
from data.dataset.torch import DecklistDataset
from loss.torch.detection_loss import v8DetectionLoss
from models.tf import EmbeddingModel
from models.torch import Detector


def make_adamw(model, lr=1e-4, momentum=0.9, decay=0.01):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    for param_name, param in model.named_parameters():
        if "bias" in param_name:  # bias (no decay)
            g[2].append(param)
        elif isinstance(param, bn):  # weight (no decay)
            g[1].append(param)
        else:  # weight (width decay)
            g[0].append(param)

    optimizer = torch.optim.AdamW(
        g[2],
        lr=lr,
        betas=(momentum, 0.999),
        weight_decay=0.0,
    )
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

    return optimizer


def make_dataset(df_path, deck_size):
    X_train = pd.read_csv(df_path)
    id_list = X_train["id"].tolist()
    prefix = "datasets/card_images_small/"
    id_list = list(map(lambda x: prefix + str(x) + ".jpg", id_list))
    card_type = list(
        map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
    )
    return DecklistDataset(id_list, card_type, deck_size)


def run_one_epoch(
    epoch,
    dataloader,
    student_model,
    teacher_model,
    loss_fn,
    is_train,
    optimizer,
):
    teacher_preprocess = EmbeddingPreprocessor()
    device = next(student_model.parameters()).device
    dtype = next(student_model.parameters()).dtype
    total_loss = torch.zeros(4)
    sample_count = 0

    for batch in dataloader:
        # ground-truth preprocess
        images = batch["image"]
        batch_size = images.shape[0]
        batch["bboxes"] = torch.from_numpy(batch["xywh"])
        batch["batch_idx"] = torch.from_numpy(batch["batch_idx"])
        batch["cls"] = torch.zeros(size=batch["batch_idx"].shape, dtype=dtype)

        # make ground-truth embedding
        gt_embeds = []
        for i, img in enumerate(images):
            mask = batch["batch_idx"] == i
            teacher_inputs = []
            img_width, img_height, _ = img.shape # deck recipe image

            # get indivisual card position
            xyxy_list = batch["xyxy"][mask.numpy()]
            xyxy_list[:, [0, 2]] *= img_width
            xyxy_list[:, [1, 3]] *= img_height
            xyxy_list = np.round(xyxy_list).astype(np.int32)

            # crop card image
            for xyxy in xyxy_list:
                x1, y1, x2, y2 = xyxy
                crop = img[y1:y2, x1:x2, :]
                crop = teacher_preprocess.resize(crop)
                teacher_inputs.append(crop)
            teacher_inputs = tf.stack(teacher_inputs)
            teacher_embeds = teacher_model(teacher_inputs).numpy()
            teacher_embeds = torch.from_numpy(teacher_embeds)
            gt_embeds.append(teacher_embeds)
        batch["embedding"] = torch.concatenate(gt_embeds, dim=0)

        # student preprocess
        student_inputs = detector_preprocessing(images).to(device)

        # inference
        student_model.train()
        if is_train:
            preds = student_model(student_inputs)
        else:
            with torch.no_grad():
                preds = student_model(student_inputs)

        # loss
        embed_topk = epoch // 50 + 1
        loss, loss_item, fg_mask = loss_fn(preds, batch, embed_topk=embed_topk)
        total_loss += loss_item.detach().cpu()
        sample_count += batch_size

        # back propagation
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # norm_loss = grad_norm(loss_item[[0, 1, 3]])
            # print(grad_norm.weights)
            optimizer.step()

    return total_loss / sample_count


def main():
    # variables
    use_logger = True
    batch_size = 8
    epochs = 100
    train_embedding_only = False
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log = logger.TrainLogger() if use_logger else None

    # prepare datasets
    train_dataset = DecklistDataset.load_from_csv("datasets/train.csv", (1, 4))
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collate_fn
    )

    # valid_dataset = make_dataset("datasets/valid.csv", 1)
    # valid_loader = DataLoader(
    #     valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn
    # )

    # prepare student model
    # pretrained_model = YOLO("weights/yolov8n_detector.pt")
    # pre_model_dict = pretrained_model.model.model.state_dict()
    pre_model_dict = torch.load("save/best.pt")

    student_model = Detector()
    model_dict = student_model.state_dict()

    # from checkpoint
    # pre_model_dict = torch.load("save/last.pt")
    # student_model.load_state_dict(pre_model_dict)
    
    for name, param in student_model.named_parameters():
        if train_embedding_only:
            if "embedding_layers" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            param.requires_grad_(False if "dfl" in name else True)

    # load pretrained yolo
    for key in pre_model_dict.keys():
        if "embedding" in key:
            continue
        model_dict[key] = pre_model_dict[key].clone().detach()
    student_model.load_state_dict(model_dict)

    student_model = student_model.cuda()
    del pre_model_dict

    # prevent TF model occupies all memories
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            raise e

    # prepare teacher model
    teacher_model = EmbeddingModel()
    teacher_model.load("embedding/weights/best.h5")

    # losses
    # box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
    det_loss = v8DetectionLoss(head=student_model.layer22, device=device, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, embed_gain=0)
    optimizer = AdamW(student_model.parameters(), lr=1e-5, weight_decay=0.01)

    # grad norm
    # grad_norm = gradNorm.GradNorm(3, student_model.layer21)

    # training
    best_loss = torch.inf
    for epoch in range(epochs):
        train_dataset.shuffle()
        train_loss = run_one_epoch(
            epoch,
            train_loader,
            student_model,
            teacher_model,
            det_loss,
            True,
            optimizer,
        )

        if use_logger:
            log.info(f"epoch: {epoch}")
            log.info("train loss")
            log.info(
                f"det_loss: {train_loss[0]} cls_loss: {train_loss[1]} dfl_loss: {train_loss[2]} embed_loss: {train_loss[3]}"
            )

        gc.collect()
        torch.cuda.empty_cache()

        # valid_loss = run_one_epoch(
        #     valid_loader,
        #     student_model,
        #     teacher_model,
        #     det_loss,
        #     False,
        #     None,
        # )
        # if use_logger:
        #     log.info(f"valid loss")
        #     log.info(
        #         f"det_loss: {valid_loss[0]} cls_loss: {valid_loss[1]} dfl_loss: {valid_loss[2]} embed_loss: {valid_loss[3]}"
        #     )

        gc.collect()
        torch.cuda.empty_cache()

        loss = train_loss.sum().detach().cpu().item()
        if loss < best_loss:
            torch.save(student_model.state_dict(), f"best.pt")
            best_loss = loss
        torch.save(student_model.state_dict(), f"last.pt")


if __name__ == "__main__":
    main()
