import torch
from functools import partial
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import numpy as np

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor, max_iter: int):
    MAX_ITER = max_iter
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3

    image = np.array(VF.to_pil_image(image_tensor))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def _apply_crf(tup, max_iter):
    return dense_crf(tup[0], tup[1], max_iter=max_iter)

def do_crf(pool, img_tensor, prob_tensor, max_iter=10):
    outputs = pool.map(partial(_apply_crf, max_iter=max_iter), zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)