import pandas as pd
from torch.utils.data import DataLoader

from utils import db, models, dataset, common


def get_catd_info(embeds, topk=1):
    pass

def make_dataset(df_path):
    X_train = pd.read_csv(df_path)
    id_list = X_train["id"].tolist()
    prefix = "datasets/card_images_small/"
    id_list = list(map(lambda x: prefix + str(x) + ".jpg", id_list))
    card_type = list(
        map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
    )
    return dataset.DecklistDataset(id_list, card_type)

def main():
    # variabels
    batch_size = 8
    chroma_db = db.ChromaDB()

    # load model
    student_model = models.Detector()
    student_model.load("best.pt")
    student_model.eval()

    # valid_dataset
    valid_dataset = make_dataset("datasets/valid.csv")
    valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn
    )

    total_image_count = 0
    for val_data in valid_loader:
        student_inputs = common.detector_preprocessing(val_data["image"])
        pred_det, pred_embed = student_model(student_inputs)
        input_shape = student_inputs.shape[1:3]
        det_results = student_model.postprocess(pred_det, pred_embed, input_shape, val_data["ori_image"])

        for det_result in det_results:
            card_names = []
            card_ids = []
            for embed in det_result.embeds:
                res = chroma_db.search_by_embed(embed.tolist())
                card_names.append(res[0][0]["name"])
                card_ids.append(res[0][0]["id"])
            det_result.names = card_names
            det_result.ids = card_ids
            det_result.save(f"valid_{total_image_count}.png")
            total_image_count += 1


if __name__ == "__main__":
    main()