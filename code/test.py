import cv2
import random
import numpy as np

import torch
from alphaiou_o import bbox_alpha_iou
from alphaiou_mm import alphaiou_loss


def gen_bbox(seed=None):
    """
    Random generate bounding boxes for IOU caculation
    Args:
        seed[int]: random seed
    Returns:
        bbox1[np.ndarray]: x1y1x2y2
        bbox2[np.ndarray]: x1y1x2y2
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # step1 : random gen bbox1
    xcyc = np.random.randint(0, 300, size=(2,))
    wh = np.random.randint(100, 320, size=(2,))
    bbox1 = [ 
                xcyc[0] - wh[0]//2,
                xcyc[1] - wh[1]//2,
                xcyc[0] + wh[0]//2,
                xcyc[1] + wh[1]//2
            ]
    bbox1 = [x.item() for x in bbox1]
    bbox1_arr = np.array(bbox1, dtype='int')

    # step2 : choose which point of bbox2 land in bbox1
    # points orderd by clockwise and 0 represent lt point of bbox1
    which_point = random.randint(0,3)
    xn = random.randint(bbox1[0], bbox1[2])
    yn = random.randint(bbox1[1], bbox1[3])
    w, h = [random.randint(32, 320) for _ in range(2)]
    if which_point == 0:
        bbox2 = [xn, yn, xn + w, yn + h]
    elif which_point == 1:
        bbox2 = [xn - w, yn, xn, yn + h]
    elif which_point == 2:
        bbox2 = [xn - w, yn - h, xn, yn]
    elif which_point == 3:
        bbox2 = [xn, yn -h, xn + w, yn]
    bbox2_arr = np.array(bbox2, dtype='int')
    
    return bbox1_arr, bbox2_arr


def vis(bbox1, bbox2):
    min_v = min(bbox1.min().item(), bbox2.min().item()) - 10
    max_v = max(bbox1.max().item(), bbox2.max().item()) + 10
    bbox1 = bbox1 - min_v
    bbox2 = bbox2 - min_v
    hw = max_v - min_v
    canvas = np.ones((hw, hw, 3), dtype=np.uint8)*255

    cv2.rectangle(canvas, tuple(bbox1[:2]), tuple(bbox1[2:]), (0, 255, 0), -1)
    cv2.rectangle(canvas, tuple(bbox2[:2]), tuple(bbox2[2:]), (0, 0, 255), -1)
    bbox3 = [
        max(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3])
        ]
    bbox3 = np.array([x.item() for x in bbox3])
    cv2.rectangle(canvas, tuple(bbox3[:2]), tuple(bbox3[2:]), (0, 128, 128), -1)

    return canvas


def main():
    bbox1, bbox2 = gen_bbox()
    canvas = vis(bbox1, bbox2)
    cv2.imwrite('vis.png', canvas)
    bbox1 = torch.from_numpy(bbox1)
    bbox2 = torch.from_numpy(bbox2)
    aiou_o = bbox_alpha_iou(bbox1, bbox2, x1y1x2y2=True, alpha=3, GIoU=True)
    print("alpha-iou original:", 1 - aiou_o)
    bbox1 = bbox1.unsqueeze(0)
    bbox2 = bbox2.unsqueeze(0)
    aiou_loss_mm = alphaiou_loss(bbox1, bbox2, alpha=3, mode='giou')
    print("alpha-iou mmdet:", aiou_loss_mm)


if __name__ == "__main__":
    main()