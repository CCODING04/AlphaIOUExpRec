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
    canvas = np.ones((hw, hw, 3), dtype=np.uint8)*128

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


def single_test():
    # only one box test
    bbox1, bbox2 = gen_bbox()
    canvas = vis(bbox1, bbox2)
    cv2.imwrite('assets/vis.png', canvas)
    bbox1 = torch.from_numpy(bbox1)
    bbox2 = torch.from_numpy(bbox2)
    aiou_o = bbox_alpha_iou(bbox1.T, bbox2, x1y1x2y2=True, alpha=3, GIoU=True)
    print("alpha-iou original:", 1 - aiou_o)
    bbox1 = bbox1.unsqueeze(0)
    bbox2 = bbox2.unsqueeze(0)
    aiou_loss_mm = alphaiou_loss(bbox1, bbox2, alpha=3, mode='giou')
    print("alpha-iou mmdet:", aiou_loss_mm)


def batch_test(num_rounds=100, mode='iou'):
    # bach cases test
    mode = mode.lower()
    assert mode in ['iou', 'giou', 'diou', 'ciou']
    all_test_results = []
    for ix in range(num_rounds):
        bboxes1, bboxes2 = list(), list()
        num_bbox = 10 # random.randint(1, 20)
        for seed in range(num_bbox):
            bbox1, bbox2 = gen_bbox(seed)
            bboxes1.append(bbox1)
            bboxes2.append(bbox2)

        # alphaiou original
        bboxes1 = np.concatenate([np.expand_dims(x, 0) for x in bboxes1], axis=0)
        bboxes2 = np.concatenate([np.expand_dims(x, 0) for x in bboxes2], axis=0)
        bboxes1 = torch.from_numpy(bboxes1)
        bboxes2 = torch.from_numpy(bboxes2)
        param_dict = {
            'box1': bboxes1.T,
            'box2': bboxes2,
            'x1y1x2y2': True,
            'GIoU': True if mode == 'giou' else False, 
            'DIoU': True if mode == 'diou' else False, 
            'CIoU': True if mode == 'ciou' else False, 
            'alpha': 3
        }
        aiou_o = bbox_alpha_iou(**param_dict)
        # results_o = []
        # for bbox1, bbox2 in zip(bboxes1, bboxes2):
        #     bbox1 = torch.from_numpy(bbox1)
        #     bbox2 = torch.from_numpy(bbox2)
        #     aiou_o = bbox_alpha_iou(bbox1, bbox2, x1y1x2y2=True, alpha=3)
        #     results_o.append(1 - aiou_o)
        aiou_loss_o = 1 - aiou_o
        results_o = aiou_loss_o.cpu().numpy().tolist()

        # alphaiou mm-implementation
        
        # bboxes1 = np.concatenate([np.expand_dims(x, 0) for x in bboxes1], axis=0)
        # bboxes2 = np.concatenate([np.expand_dims(x, 0) for x in bboxes2], axis=0)
        # bboxes1 = torch.from_numpy(bboxes1)
        # bboxes2 = torch.from_numpy(bboxes2)
        aiou_loss_mm = alphaiou_loss(bboxes1, bboxes2, alpha=3, mode=mode)
        results_mm = aiou_loss_mm.cpu().numpy().tolist()
        
        
        """
        print(f'======{ix} round test======')
        # print('original loss:', results_o)
        # print('mm-imp loss:', results_mm)
        is_eq = all([x==y for x, y in zip(results_o, results_mm)])
        all_test_results.append(is_eq)
        print("test results [original loss] == [mm-imp loss]",is_eq)
        """
        # print(aiou_loss_o)
        # print(aiou_loss_mm)
        is_eq = all([x==y for x, y in zip(results_o, results_mm)])
        all_test_results.append(is_eq)
        
    print(f'# Tests {mode.upper()}: {num_rounds}')
    print('All tests equal:', all(all_test_results))
        



def main():
    # torch.set_printoptions(precision=10)
    torch.set_printoptions(sci_mode=True)
    # single_test()
    for mode in ['iou', 'giou', 'diou', 'ciou']:
        batch_test(mode=mode)



if __name__ == "__main__":
    main()