import math
import torch

def alphaiou_loss(pred, target, alpha=3, eps=1e-9, mode='iou'):
    # iou mode
    mode = mode.lower()
    assert mode in ('iou', 'ciou', 'giou', 'diou')

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    # overlap
    overlap = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) *\
              (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # union

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - overlap + eps
    

    # change conventional iou to alpha pow
    ious = torch.pow(overlap / union + eps, alpha)
    
    # calculate alpha-iou according mode
    if mode == 'iou':
        loss = 1 - ious
        return loss

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    if mode == 'giou':
        c_area = torch.max(cw * ch + eps, union)
        gious = ious - torch.pow((c_area - union) / c_area + eps, alpha)
        loss = 1 - gious
        return loss

    c2 = (cw**2 + ch**2)**alpha + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = (left + right)**alpha

    # DIoU
    if mode == 'diou':
        dious = ious - rho2 / c2
        loss = 1 - dious
        return loss
    else:
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        factor = 4 / math.pi**2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha_ciou = (ious > 0.5).float() * v / (1 - ious + v)

        # CIoU
        cious = ious - (rho2 / c2 + torch.pow(alpha_ciou * v, alpha))
        loss = 1 - cious
        return loss
