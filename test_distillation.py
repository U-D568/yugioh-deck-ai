import pandas as pd
from torch.utils.data import DataLoader
from ultralytics import YOLO

from utils import db, models, dataset, common
from utils.models import Detector, EmbeddingModel


def make_dataset(df_path):
    X_train = pd.read_csv(df_path)
    id_list = X_train["id"].tolist()
    prefix = "datasets/card_images_small/"
    id_list = list(map(lambda x: prefix + str(x) + ".jpg", id_list))
    card_type = list(
        map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
    )
    return dataset.DecklistDataset(id_list, card_type, 1)


def main():
    # variabels
    batch_size = 8
    chroma_db = db.ChromaDB()

    # load model
    student_model = models.Detector()
    student_model.load("save/last.pt")

    # from scratch
    # student_model = models.Detector()
    # model_dict = student_model.state_dict()
    # pretrained_model = YOLO("weights/yolov8n_detector.pt")
    # pre_model_dict = pretrained_model.model.model.state_dict()

    # for key in pre_model_dict.keys():
    #     new_key = f"layer{key}"
    #     model_dict[new_key] = pre_model_dict[key].clone().detach()
    # student_model.load_state_dict(model_dict)

    student_model = student_model.cuda()
    student_model.eval()

    # valid_dataset
    valid_dataset = make_dataset("datasets/valid.csv")
    valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn
    )

    total_image_count = 0
    student_embeddings = []
    for val_data in valid_loader:
        student_inputs = common.detector_preprocessing(val_data["image"])
        pred_det, pred_embed = student_model(student_inputs.cuda())
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
                res = chroma_db.search_by_embed(embed.tolist())
                card_names.append(res[0][0]["name"])
                card_ids.append(res[0][0]["id"])
            det_result.names = card_names
            det_result.ids = card_ids
            det_result.save(f"valid_{total_image_count}.png")
            total_image_count += 1
        exit()


if __name__ == "__main__":
    main()
