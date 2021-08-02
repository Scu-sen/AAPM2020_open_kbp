import numpy as np
import torch
from scipy import ndimage
from albumentations import Compose, to_tuple
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, to_tuple

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.filterwarnings('ignore', '.*This overload of addcmul_*')


def get_train_tfms(config):
    tfms = []
    if config.horizontalflip is not None: tfms.append(HorizontalFlip(**config.horizontalflip))
    if config.randomrotate90 is not None: tfms.append(RandomRotate90(**config.randomrotate90))
    if config.randomscale is not None: tfms.append(RandomScale(**config.randomscale))
    if config.randomrotate is not None: tfms.append(RandomRotate(**config.randomrotate))
    if config.randomshift is not None: tfms.append(RandomShift(**config.randomshift))
    if config.noise_multiplier is not None: tfms.append(MultiplicativeNoise(**config.noise_multiplier))
    if config.gaussianblur is not None: tfms.append(GaussianBlur(**config.gaussianblur))
    if config.randomcrop is not None: tfms.append(RandomCrop(**config.randomcrop))
    if config.verticalflip is not None: tfms.append(VerticalFlip(**config.verticalflip))
    return Compose(tfms)

class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """
    def __init__(self, axis=3, always_apply=False, p=0.5):
        super(HorizontalFlip, self).__init__(always_apply, p)
        self.axis = axis
    
    def apply(self, img, **params):
        assert len(img.shape) == 3
        if type(img) == np.ndarray:
            if self.axis == 3:
                img = np.ascontiguousarray(img[:, :, ::-1])
            elif self.axis == 1:
                img = np.ascontiguousarray(img[:, ::-1, :])
        else:
            revidx = torch.arange(127,-1,-1)
            if self.axis == 3:
                img = img[:, :, revidx]
            elif self.axis == 1:
                img = img[:, revidx, :]
        return img

    def get_transform_init_args_names(self):
        return ()

class VerticalFlip(DualTransform):
    """Flip the input vertically.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """
    def __init__(self, num_channels=108, always_apply=False, p=0.5):
        super(VerticalFlip, self).__init__(always_apply, p)
        self.num_channels = num_channels
    
    def apply(self, img, **params):
        assert len(img.shape) == 3
        flipidxs = np.arange(img.shape[0])
        flipidxs[:self.num_channels] = flipidxs[:self.num_channels][::-1]
        for i in range(0, img.shape[0], 12):
            flipidxs[i:i+12] = flipidxs[i:i+12][::-1]
        img = np.ascontiguousarray(img[flipidxs])
        return img

    def get_transform_init_args_names(self):
        return ()

class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is same as the input image size.
    Args:
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
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
            scale_limit = [scale_limit, scale_limit]
        self.scale_limit = [to_tuple(sl, bias=1.0) for sl in scale_limit]
        self.interpolation = interpolation

    def get_params(self):
        return {"scalex": np.random.uniform(self.scale_limit[0][0], self.scale_limit[0][1]),
                "scaley": np.random.uniform(self.scale_limit[1][0], self.scale_limit[1][1]),
                "order": self.interpolation}

    def apply(self, img, scalex=1, scaley=1, order=0, **params):
        assert len(img.shape) == 3
        oz = ndimage.zoom(img, (1, scalex, scaley), order=order, mode='nearest')

        # Pad with zeros
        pad = (img.shape - np.array(oz.shape)+1)//2
        ozxind = np.s_[:] if pad[1] < 0 else np.s_[pad[1]:pad[1]+oz.shape[1]]
        ozyind = np.s_[:] if pad[2] < 0 else np.s_[pad[2]:pad[2]+oz.shape[2]]
        ozf = np.zeros(np.maximum(img.shape, oz.shape), dtype=img.dtype)
        ozf[:,ozxind,ozyind] = oz

        # Crop to original size
        crop = (ozf.shape - np.array(img.shape))//2
        ozxind = np.s_[:img.shape[1]] if crop[1] <= 0 else np.s_[crop[1]:crop[1]+img.shape[1]]
        ozyind = np.s_[:img.shape[2]] if crop[2] <= 0 else np.s_[crop[2]:crop[2]+img.shape[2]]
        return ozf[:,ozxind,ozyind]

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit[0], bias=-1.0)[1]}


class RandomShift(DualTransform):
    """Randomly shift the input. Output image size is same as the input image size.
    Args:
        shift_limit ((int, int) or int): shift range in pixels. If shift_limit is a single float value, the
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
            shift_limit = [shift_limit, shift_limit]
        self.shift_limit = [to_tuple(sl, bias=0) for sl in shift_limit]
        self.interpolation = interpolation

    def get_params(self):
        return {"shiftx": np.random.randint(self.shift_limit[0][0], self.shift_limit[0][1]+1),
                "shifty": np.random.randint(self.shift_limit[1][0], self.shift_limit[1][1]+1),
                "order": self.interpolation}

    def apply(self, img, shiftx=0, shifty=0, order=0, **params):
        img = ndimage.interpolation.shift(img, (0, shiftx, shifty), mode='nearest', order=order)
        return img

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "shift_limit": to_tuple(self.shift_limit[0], bias=0)[1]}

class RandomCrop(DualTransform):
    """Randomly center crop the input. Output image size is same as the input image size.
    Args:
        crop_limit ((float, float) or float): maximum size of the crop in x and y. If a single float value, same for both dimensions. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.75.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, crop_limit=0.1, always_apply=False, p=0.75):
        super(RandomCrop, self).__init__(always_apply, p)
        if type(crop_limit) == float or type(crop_limit) == int:
            crop_limit = [crop_limit, crop_limit]
        self.crop_limit = crop_limit

    def get_params(self):
        cropx = np.random.uniform(self.crop_limit[0], 1)
        cropy = np.random.uniform(self.crop_limit[1], 1)
        return {"cropx": cropx, "cropy": cropy}

    def apply(self, img, cropx=1, cropy=1, **params):
        _, h, w = np.array(img.shape)
        cropx_px = (h * cropx).astype('int')
        cropy_px = (w * cropy).astype('int')
        mask = np.zeros_like(img)
        mask[:, (h-cropx_px)//2:(h+cropx_px)//2+1, (w-cropy_px)//2:(w+cropy_px)//2+1] = 1
        img *= mask
        return img

    def get_transform_init_args(self):
        return {"crop_limit": self.crop_limit}

class RandomRotate(DualTransform):
    """Randomly rotate the input. Output image size is same as the input image size.
    Args:
        max_angle (float): maximum rotation angle in degrees. If max_angle is a single float value, the
            range will be (-max_angle, max_angle). Default: (-2.0, 2.0).
        p (float): probability of applying the transform. Default: 0.25.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, max_angle=2.0, interpolation=0, always_apply=False, p=0.25):
        super(RandomRotate, self).__init__(always_apply, p)
        self.max_angle = to_tuple(max_angle, bias=0)
        self.interpolation = interpolation

    def get_params(self):
        return {"angle": np.random.uniform(self.max_angle[0], self.max_angle[1]),
                "order": self.interpolation}

    def apply(self, img, angle=0, order=0, **params):
#         print('angle={}'.format(angle))
        assert len(img.shape) == 3
        rotated_img = ndimage.interpolation.rotate(img, angle, axes=(1, 2), mode='nearest', order=order, reshape=False)
        return rotated_img

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "max_angle": to_tuple(self.max_angle[0], bias=0)[1]}

class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees one or more times.
    Args:
        p (float): probability of applying the transform. Default: 0.25.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=0.25):
        super(RandomRotate90, self).__init__(always_apply, p)

    def get_params(self):
        return {"times": np.random.choice([1, 2, 3])}

    def apply(self, img, times, **params):
        assert len(img.shape) == 3
        rotated_img = np.ascontiguousarray(np.rot90(img, k=times, axes=(1,2)))
        return rotated_img

    def get_transform_init_args(self):
        return { }

class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image to random number or array of numbers.
    Args:
        multiplier (float or tuple of floats): If single float image will be multiplied to this number.
            If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
        per_channel (bool): If `False`, same values for all channels will be used.
            If `True` use sample values for each channels. Default False.
        elementwise (bool): If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.
    Targets:
        image
    Image types:
        Any
    """

    def __init__(self, multiplier=(0.99, 1.01), elementwise=False, always_apply=False, p=0.5):
        super(MultiplicativeNoise, self).__init__(always_apply, p)
        self.multiplier = multiplier
        self.elementwise = elementwise

    def apply(self, img, multiplier, imgch, **kwargs):
        assert img.shape[1:] == (128, 128) and len(img.shape) == 3
        img[imgch] *= multiplier
        return img

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        imgch = np.where(img.max(1).max(1) > 200)
        if self.elementwise:
            shape = (len(imgch), img.shape[1], img.shape[2])
        else:
            shape = [1]
        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1], shape).astype(img.dtype)
        return {"multiplier": multiplier, "imgch": imgch}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "multiplier", "elementwise"

class GaussianBlur(ImageOnlyTransform):
    """Apply gaussian blur by a random std. Only operates on the image channels
    Args:
        sigma_limit (float): sigma will be chosen from (0., sigma). Default 1.
        blur_channels (bool): Blur across image channels. Default False.
    Targets:
        image
    Image types:
        Any
    """

    def __init__(self, sigma_limit=1.0, blur_channels=False, always_apply=False, p=0.5):
        super(GaussianBlur, self).__init__(always_apply, p)
        self.sigma_limit = sigma_limit
        self.blur_channels = blur_channels

    def get_params(self):
        sigma = np.random.uniform(0, self.sigma_limit)
        sigma = [0, sigma, sigma]
        if self.blur_channels:
            sigma = [sigma, sigma, sigma]
        return {"sigma": sigma}
    
    def apply(self, img, sigma, **kwargs):
        assert img.shape[1:] == (128, 128) and len(img.shape) == 3
        img_ch = np.where(img.max(1).max(1) > 150)
        img[img_ch] = ndimage.gaussian_filter(img[img_ch], sigma, mode='constant')
        return img

    def get_transform_init_args(self):
        return {"sigma_limit": self.sigma_limit, "blur_channels": self.blur_channels}

class Presize(DualTransform):
    """Randomly zoom into a part of the input.
    Args:
        zoom_limit (int): the maximum zoom that can be applied
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, zoom_limit, interpolation=0, always_apply=True, p=1):
        super(Presize, self).__init__(always_apply, p)
        assert zoom_limit >= 1, 'Zoom limit should be greater than 1'
        self.zoom_limit = zoom_limit
        self.interpolation = interpolation
    
    def get_params(self):
        new_size = np.random.uniform(1./self.zoom_limit, 1.)
        top = np.random.uniform(0, 1 - new_size)
        left = np.random.uniform(0, 1 - new_size)
        return {"new_size": new_size, "top": top, "left": left}

    def apply(self, img, new_size, top, left, **params):
        assert len(img.shape) == 3
        out = np.zeros_like(img)
        new_size_px = np.round(img.shape[1]*new_size).astype('int')
        top_px = np.round(img.shape[1]*top).astype('int')
        left_px = np.round(img.shape[2]*left).astype('int')
        img = img[:, top_px:top_px+new_size_px, left_px:left_px+new_size_px]
        
        # Zoom back to original size
        scale = out.shape[1]/img.shape[1]
        for i in range(img.shape[0]):
            out[i] = ndimage.zoom(img[i], scale, order=self.interpolation, mode='nearest')
        return out

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "zoom_limit": self.zoom_limit}

class Diffeomorph(DualTransform):
    """Apply diffeomorphic transformation to the input.
    Args:
        iter_limits (list of lists): the limits of optimization iterations at every gaussian pyramid level
        interpolation: 'nearest' or 'linear'
        metric: 'SSD', others not implemented (SSDMetric, CCMetric, EMMetric)
        mode: 'adjacent', 'flip': adjacent gets morphing between randomly selected adjacent slides, flip gets morphing between a randomly selected image and its flip
        p (float): probability of applying the transform. Default: 0.75.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, iter_limits=[[64, 32, 8, 2], [128, 64, 32, 8]], interpolation='nearest', metric='SSD', mode='flip', always_apply=False, p=0.75):
        super(Diffeomorph, self).__init__(always_apply, p)
        self.iter_limits = iter_limits
        self.interpolation = interpolation
        if metric == 'SSD':
            self.metric = SSDMetric(dim=2)
        else:
            raise NotImplementedError
        self.mode = mode

    def get_params(self):
        level_iters = [np.random.randint(i, j) for i, j in zip(self.iter_limits[0], self.iter_limits[1])]
        sdr = SymmetricDiffeomorphicRegistration(self.metric, level_iters, inv_iter=50)
        return {"sdr": sdr}

    def apply(self, img, sdr, **params):
        assert len(img.shape) == 3
        nimgs = (img.shape[0] // 12) - 1
        if self.mode == 'adjacent':
            imgch = np.random.randint(0, nimgs - 1)
            static = img[imgch*12]
            moving = img[(imgch+1)*12]
            if np.random.rand() > 0.5:
                static, moving = moving, static
        elif self.mode == 'flip':
            imgch = np.random.randint(0, nimgs)
            moving = img[imgch*12]
            static = moving[:,::-1]
        else:
            raise ValueError("Please select one of two defined modes")

        if static.sum() == 0 or moving.sum() == 0:
            return img

        mapping = sdr.optimize(static, moving)
        out = np.zeros_like(img)
        for i in range(img.shape[0]):
            out[i] = mapping.transform(img[i], self.interpolation)
        return out

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "iter_limits": self.iter_limits}

class MonaiAffine(DualTransform):
    """Apply affine transformation from MONAI library
    Args:
        max_angle (angle in radians): the rotation will be applied from (-max_angle, max_angle)
        shear_limit (maximum allowed shear): float
        scale_limit (lower limit and upper limit of scaling): tuple (float)
        shift_limit (pixels by which to shift the image): float
        interpolation: 'nearest' or 'linear'
        p (float): probability of applying the transform. Default: 0.75.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, max_angle=np.pi/8, shear_limit=0.15, scale_limit=(0.8, 1.2), shift_limit=5, interpolation='nearest', always_apply=False, p=0.75):
        super(MonaiAffine, self).__init__(always_apply, p)
        self.max_angle = max_angle
        self.shear_limit = shear_limit
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit
        self.interpolation = interpolation
        from monai.transforms import Affine
        self.Affine = Affine

    def get_params(self):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        shearx = np.random.uniform(-self.shear_limit, self.shear_limit)
        sheary = np.random.uniform(-self.shear_limit, self.shear_limit)
        scalex = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        scaley = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        shiftx = np.random.uniform(-self.shift_limit, self.shift_limit)
        shifty = np.random.uniform(-self.shift_limit, self.shift_limit)
        affine = self.Affine(rotate_params=angle, shear_params=(shearx, sheary), scale_params=(scalex, scaley), translate_params=(shiftx, shifty), padding_mode='zeros', as_tensor_output=False)
        return {"affine": affine}

    def apply(self, img, affine, **params):
        assert len(img.shape) == 3
        return affine(img, img.shape[1:], mode=self.interpolation)

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "max_angle": self.max_angle, "shear_limit": self.shear_limit, "scale_limit": self.scale_limit, "shift_limit": self.shift_limit}

class MonaiElastic(DualTransform):
    """Apply elastic transformation from MONAI library
    Args:
        interpolation: 'nearest' or 'linear'
        p (float): probability of applying the transform. Default: 0.75.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, interpolation='nearest', always_apply=False, p=0.5):
        super(MonaiElastic, self).__init__(always_apply, p)
        self.interpolation = interpolation
        from monai.transforms import Affine, Rand2DElastic
        self.Affine = Affine
        self.Rand2DElastic = Rand2DElastic
        self.affine = self.Affine(scale_params=(0.925, 0.925), padding_mode='zeros', as_tensor_output=False)
        self.elastic = self.Rand2DElastic(prob=1.0, spacing=(2, 2), magnitude_range=(0, 1), rotate_range=(0), scale_range=(0., 0.), translate_range=(0, 0), padding_mode='zeros', mode=self.interpolation)

    def get_params(self):
        return {}

    def apply(self, img, **params):
        assert len(img.shape) == 3
        imshape = img.shape[1:]
        return self.affine(self.elastic(img, imshape, mode=self.interpolation), imshape, mode=self.interpolation)

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation}
