import time

import pandas as pd
from torch.utils.data import DataLoader
from ultralytics import YOLO

import cv2
import db
from utils import common
import torch
from data.preprocess.torch import detector_preprocessing
from data.dataset.torch import DecklistDataset
from models.torch import Detector
from models.tf import EmbeddingModel


def main():
    # variabels
    batch_size = 8
    chroma_db = db.ChromaDBConnection()

    # load model
    student_model = Detector()
    student_model.load("best.pt")

    # from scratch
    # student_model = models.Detector()
    # model_dict = student_model.state_dict()
    # pretrained_model = YOLO("weights/yolov8n_detector.pt")
    # pre_model_dict = pretrained_model.model.model.state_dict()

    # for key in pre_model_dict.keys():
    #     new_key = f"layer{key}"
    #     model_dict[new_key] = pre_model_dict[key].clone().detach()
    # student_model.load_state_dict(model_dict)

    # student_model = student_model.cuda()
    student_model.eval()
    
    # teacher_model = EmbeddingModel()
    # teacher_model.load("embedding/weights/best.h5")

    # valid_dataset
    valid_dataset = DecklistDataset.load_from_csv("datasets/train.csv", 1)
    valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn
    )

    total_image_count = 0

    top1_acc = 0
    top5_acc = 0
    total_instance_count = 0
    pred_test = []
    gt_test = []
    for val_data in valid_loader:
        student_inputs = detector_preprocessing(val_data["image"])
        with torch.no_grad():
            # pred_det, pred_embed = student_model(student_inputs.cuda())
            pred_det, pred_embed = student_model(student_inputs)
        if len(student_inputs.shape) == 4:
            input_shape = student_inputs.shape[2:4]
        else:
            input_shape = student_inputs.shape[1:3]
        det_results = student_model.postprocess(
            pred_det, pred_embed, input_shape, val_data["ori_image"]
        )

        for det_result in det_results:
            card_names = []
            card_ids = []
            for embed in det_result.embeds:
                # student_embeddings.append(embed)
                res = chroma_db.search_by_embed(embed.tolist(), 20)[0]
                names = list(map(lambda x: x["name"], res))
                ids = list(map(lambda x: x["id"], res))
                card_names.append(names)
                card_ids.append(ids)
            det_result.names = card_names
            det_result.ids = card_ids
            # det_result.save(f"valid_{total_image_count}.png")
            total_image_count += 1

        for gt_ids, det_result in zip(val_data["ids"], det_results):
            total_instance_count += len(gt_ids)
            for gt_id in gt_ids:
                gt_id = int(gt_id)
                for pred_id in det_result.ids:
                    if gt_id in pred_id:
                        top5_acc += 1
                    else:
                        pred_test.append(pred_id)
                        gt_test.append(gt_id)
                    if int(gt_id) == pred_id[0]:
                        top1_acc += 1
    print(f"top1 acc: {top1_acc/total_instance_count*100}%")
    print(f"top5 acc: {top5_acc/total_instance_count*100}%")
    print(1)


if __name__ == "__main__":
    main()
