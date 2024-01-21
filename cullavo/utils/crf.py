import torch
import torchvision.transforms.functional as VF
import numpy as np

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def dense_crf(image_tensor: torch.FloatTensor, output_probs: torch.FloatTensor, max_iter: int):
    MAX_ITER = max_iter
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 1
    Bi_XY_STD = 80
    Bi_RGB_STD = 13

    image = np.array(VF.to_pil_image(image_tensor))[:, :, ::-1]
    image = np.ascontiguousarray(image)

    output_probs = torch.stack([1-output_probs, output_probs]).cpu().numpy()
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

def apply_crf(image, logit, max_iter):
    outputs = dense_crf(image.cpu(), logit.float().detach().cpu(), max_iter=max_iter)
    return outputs