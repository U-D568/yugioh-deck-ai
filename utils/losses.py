import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


def cosine_distance(x1: tf.Tensor, x2: tf.Tensor):
    inner_products = tf.tensordot(x1, x2, axes=-1)
    x1_norm = tf.norm(x1, ord="euclidean", axis=-1)
    x2_norm = tf.norm(x2, ord="euclidean", axis=-1)
    return 1 - (inner_products / (x1_norm * x2_norm))


def contrastive_loss(anchor, pred, y, margin=0.5):
    distances = tf.math.reduce_euclidean_norm(anchor - pred, axis=-1)
    margin_distances = tf.maximum(margin - distances, 0)
    losses = (1 - y) * tf.math.pow(distances, 2) + y * tf.math.pow(margin_distances, 2)
    return losses


def square_norm(x1, x2):
    return tf.math.reduce_sum(tf.math.square(x1 - x2), axis=-1)


def triplet_loss(anchor, positive, negative, margin=0.5):
    distance_positive = square_norm(anchor, positive)
    distance_negative = square_norm(anchor, negative)
    loss = tf.maximum(distance_positive - distance_negative + margin, 0)

    return loss


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(
        self, head, device, tal_topk=10, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5
    ):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""

        m = head  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = {"box": box_gain, "cls": cls_gain, "dfl": dfl_gain}
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.embedding_size = m.embedding_size
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def preprocess_embedding(self, targets, batch_size):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, use_cosine):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        pred_detect = preds[0]
        pred_embeds = preds[1]
        feats = pred_detect[1] if isinstance(pred_detect, tuple) else pred_detect
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # embeeding
        pred_embeds = torch.cat(
            [xi.view(feats[0].shape[0], self.embedding_size, -1) for xi in pred_embeds],
            dim=2,
        )
        pred_embeds = pred_embeds.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        targets_embed = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["embedding"]), dim=1
        )
        gt_embeds = self.preprocess_embedding(targets_embed.to(self.device), batch_size)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        del targets, targets_embed

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, target_embeds, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            (pred_embeds.detach()).type(gt_embeds.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_embeds,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # Embedding loss (euclidean distance)
        if use_cosine:
            pred_embeds = torch.nn.functional.normalize(pred_embeds, p=2, dim=-1)
            target_embeds = torch.nn.functional.normalize(target_embeds, p=2, dim=-1)
            cos = torch.nn.functional.cosine_similarity(target_embeds, pred_embeds, dim=-1)
            loss[3] = (1 - cos).sum() / batch_size
        else:
            loss[3] = torch.square(target_embeds - pred_embeds).sum() / batch_size

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp["dfl"]  # dfl gain

        # loss(box, cls, dfl)
        return loss.sum() * batch_size, loss.detach()
