import numpy as np
from scipy import ndimage
from albumentations.core.transforms_interface import DualTransform, to_tuple
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        img = np.ascontiguousarray(img[:, :, ::-1, :])
        return img

    def get_transform_init_args_names(self):
        return ()

def zoom_and_resize(img, scale, order):
    orig_shape = img.shape
    assert len(img.shape) == 3
    out = ndimage.zoom(img, scale, order=order, mode='nearest')
    
    #Pad with zeros
    min_pad = -min(min(np.array(out.shape) - orig_shape)-1, 0)//2
    out = np.pad(out, pad_width=min_pad)
    
    # Crop to original size
    crop_idx = (np.array(out.shape) - orig_shape)//2
    crop_idx = [np.arange(i, i+j) for i, j in zip(crop_idx, orig_shape)]
    return out[crop_idx[0]][:, crop_idx[1]][:,:,crop_idx[2]]

class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is same as the input image size.
    Args:
        scale_limit ((float, float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, scale_limit=0.1, interpolation=0, always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)
        if type(scale_limit) == float:
            scale_limit = [scale_limit, scale_limit, scale_limit]
        self.scale_limit = [to_tuple(sl, bias=1.0) for sl in scale_limit]
        self.interpolation = interpolation

    def get_params(self):
        return {"scalex": np.random.uniform(self.scale_limit[0][0], self.scale_limit[0][1]),
                "scaley": np.random.uniform(self.scale_limit[1][0], self.scale_limit[1][1]),
                "scalez": np.random.uniform(self.scale_limit[2][0], self.scale_limit[2][1]),
                "order": self.interpolation}

    def apply(self, img, scalex=1, scaley=1, scalez=1, order=0, **params):
        assert len(img.shape) == 4
        zoomed_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            zoomed_img[i] = zoom_and_resize(img[i], (scalex, scaley, scalez), order=order)
        return zoomed_img

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit[0], bias=-1.0)[1]}


class RandomShift(DualTransform):
    """Randomly shift the input. Output image size is same as the input image size.
    Args:
        shift_limit ((int, int, int) or int): shift range in pixels. If shift_limit is a single float value, the
            range will be (-shift_limit, shift_limit). Default: (-2, 2).
        p (float): probability of applying the transform. Default: 0.75.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, shift_limit=2, interpolation=0, always_apply=False, p=0.75):
        super(RandomShift, self).__init__(always_apply, p)
        if type(shift_limit) == int:
            shift_limit = [shift_limit, shift_limit, shift_limit]
        self.shift_limit = [to_tuple(sl, bias=0) for sl in shift_limit]
        self.interpolation = interpolation

    def get_params(self):
        return {"shiftx": np.random.randint(self.shift_limit[0][0], self.shift_limit[0][1]+1),
                "shifty": np.random.randint(self.shift_limit[1][0], self.shift_limit[1][1]+1),
                "shiftz": np.random.randint(self.shift_limit[2][0], self.shift_limit[2][1]+1),
                "order": self.interpolation}

    def apply(self, img, shiftx=0, shifty=0, shiftz=0, order=0, **params):
#         print('shiftx={}, shifty={}, shiftz={}'.format(shiftx, shifty, shiftz))
#         assert len(img.shape) == 4
#         if shiftx > 0:
#             img = np.concatenate((np.zeros((img.shape[0], shiftx, img.shape[2], img.shape[3])), img), axis=1)[:,:img.shape[1],:,:]
#         else:
#             shiftx *= -1
#             img = np.concatenate((img, np.zeros((img.shape[0], shiftx, img.shape[2], img.shape[3]))), axis=1)[:,shiftx:,:,:]
#         if shifty > 0:
#             img = np.concatenate((np.zeros((img.shape[0], img.shape[1], shifty, img.shape[3])), img), axis=2)[:,:,:img.shape[2],:]
#         else:
#             shifty *= -1
#             img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], shifty, img.shape[3]))), axis=2)[:,:,shifty:,:]
#         if shiftz > 0:
#             img = np.concatenate((np.zeros((img.shape[0], img.shape[1], img.shape[2], shiftz)), img), axis=3)[:,:,:,:img.shape[1]]
#         else:
#             shiftz *= -1
#             img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], img.shape[2], shiftz))), axis=3)[:,:,:,shiftz:]
        img = ndimage.interpolation.shift(img, (0, shiftx, shifty, shiftz), mode='nearest', order=order)
        return img

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "shift_limit": to_tuple(self.shift_limit[0], bias=0)[1]}

def rotate_and_resize(img, angle, order=0):
    assert len(img.shape) == 3
    anglex, angley, anglez = angle
    img = ndimage.interpolation.rotate(img, anglez, axes=(0, 1), mode='nearest', order=order, reshape=False)
    img = ndimage.interpolation.rotate(img, angley, axes=(0, 2), mode='nearest', order=order, reshape=False)
    img = ndimage.interpolation.rotate(img, anglex, axes=(1, 2), mode='nearest', order=order, reshape=False)
    return img

class RandomRotate(DualTransform):
    """Randomly rotate the input. Output image size is same as the input image size.
    Args:
        max_angle ((float, float, float) or float): maximum rotation angle in degrees. If max_angle is a single float value, the
            range will be (-max_angle, max_angle). Default: (-2.0, 2.0).
        p (float): probability of applying the transform. Default: 0.25.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, max_angle=2.0, interpolation=0, always_apply=False, p=0.25):
        super(RandomRotate, self).__init__(always_apply, p)
        if type(max_angle) == int:
            max_angle = [max_angle, max_angle, max_angle]
        self.max_angle = [to_tuple(sl, bias=0) for sl in max_angle]
        self.interpolation = interpolation

    def get_params(self):
        return {"anglex": np.random.uniform(self.max_angle[0][0], self.max_angle[0][1]),
                "angley": np.random.uniform(self.max_angle[1][0], self.max_angle[1][1]),
                "anglez": np.random.uniform(self.max_angle[2][0], self.max_angle[2][1]),
                "order": self.interpolation}

    def apply(self, img, anglex=0, angley=0, anglez=0, order=0, **params):
#         print('anglex={}, angley={}, anglez={}'.format(anglex, angley, anglez))
        assert len(img.shape) == 4
        rotated_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            rotated_img[i] = rotate_and_resize(img[i], (anglex, angley, anglez), order=order)
        return rotated_img

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "max_angle": to_tuple(self.max_angle[0], bias=0)[1]}

