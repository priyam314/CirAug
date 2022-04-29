# Import Libraries
import kornia as K

# Custom Modules
from ProAug.augOperator import AugOperator
from ProAug.augParam import AugParam
from ProAug.augSeq import AugSeq
from ProAug.param import Param
from ProAug.rangeParam import *


# Initiate AugSeq()
Augs = AugSeq()

# Define all possible Augmentation Operators
randomMotionBlurOpr = AugOperator("Random Motion Blur", K.augmentation.RandomMotionBlur, 1.0, AugParam(
						Param("kernel_size", 5, DecDiscreteRange(3, 9)),
                        Param("angle", 90, LoopContRange(1.5, 358.5)),
                        Param("direction", 3, ContRange(0.1, 3.9))))

randomPosterizeOpr = AugOperator("Random Poseterize", K.augmentation.RandomPosterize, 1.0, AugParam(
                        Param("bits", 5, DecDiscreteRange(8, 5))))

colorJiggleOpr = AugOperator("Color Jiggle", K.augmentation.ColorJiggle, 1.0, AugParam(
                        Param("brightness", 1.4, ContRange(0.1, 2.0)),
                        Param("contrast", 1.8, DecContRange(3.0, 0.2)),
                        Param("saturation", 2.6, LoopContRange(0.1, 3.0))))

randomSharpnessOpr = AugOperator("Random Sharpness", K.augmentation.RandomSharpness, 1.0, AugParam(
                        Param("sharpness", 0.5, ContRange(0.01, 0.87))))

randomErasingOpr = AugOperator("Random Erasing", K.augmentation.RandomErasing, 1.0, AugParam(
                        Param("scale", (0.25, 0.25), TupleContRange(0.1, 0.3))))

randomResizedCropOpr = AugOperator("Random Resied Crop", K.augmentation.RandomResizedCrop, 1.0, AugParam(
                        Param("scale", (0.75, 0.75), TupleDecContRange(0.99, 0.75)),
                        Param("size", (96,96), IdentityRange(96,96))))


# aspectRatioOpr = AugOperator( "change_aspect_ratio", change_aspect_ratio_resize, 1.0,
# 							AugParam( Param("ratio", 3.0, ContRange(0.01, 5.0))))

# colorJitterOpr = AugOperator( "color_jitter", color_jitter, 1.0, AugParam(
#             					Param("brightness_factor", 1.4, ContRange(0.1, 2.0)),
#             					Param("contrast_factor", 1.0, DecContRange(3.0, 0.1)),
#             					Param("saturation_factor", 1.0, LoopContRange(0.05, 5.0))))

# encodingQualityOpr = AugOperator( "encoding_quality", encoding_quality, 1.0, 
#                                  AugParam(Param("quality", 75.0, DecContRange(94.0, 9.0))))

# grayscaleOpr = AugOperator( "grayscale", grayscale, 1.0, AugParam(
#                                 Param("mode", "luminosity", EnumRange("luminosity", "average"))))
										
# hFlipOpr = AugOperator( "hflip", hflip, 1.0 )

# opacityOpr = AugOperator("opacity", opacity, 1.0, AugParam(
# 							Param("level", 1.0, DecContRange(0.15, 0.98))))

# perspectiveOpr = AugOperator("perspective_transform", perspective_transform, 1.0,
#                                  AugParam( Param("sigma", 1.0, ContRange(0.1, 20.0))))

# pixelizationOpr = AugOperator("pixelization", pixelization, 1.0, AugParam(
# 							Param("ratio", 1.0, DecContRange(0.99, 0.2))))

# randomNoiseOpr = AugOperator("random_noise", random_noise, 1.0, AugParam(
#                                 Param("var", 0.009, ContRange(0.001, 0.01))))

# rotateOpr = AugOperator("rotate", rotate, 1.0, AugParam(
#                             Param("degrees", 90, LoopContRange(0.0, 359.9))))

# shufflePixelOpr = AugOperator("shuffle_pixels", shuffle_pixels, 1.0, AugParam(
#                                 Param("factor", 0.06, ContRange(0.001, 0.1))))
Augs.add_augObj_List([
    randomMotionBlurOpr,
    randomPosterizeOpr,
    colorJiggleOpr,
    randomSharpnessOpr,
    randomErasingOpr,
    randomResizedCropOpr
])



