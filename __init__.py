import cellpose
import numpy as np
import cellpose.models
import torch
import cv2
from skimage.measure import regionprops_table
from typing import Union
from .intensity_functions import mean_intensity, mean_80_intensity

PROPERTIES = ["label", "area", "area_convex", "area_filled", "axis_major_length", "axis_minor_length", "eccentricity", "equivalent_diameter_area", "perimeter", "solidity"]

def main_function(I: 'np.ndarray[np.uint8]', 
                  intensity_image: 'np.ndarray[np.float32]'=None,
                  cellpose_net: Union['str', cellpose.models.CellposeModel] = None, 
                  eval_kwargs:dict=None, 
                  refine: bool=False, 
                  t: float=0.12, 
                  out_sz: tuple=None, 
                  extract_morph_features: bool=False, 
                  extract_intensity_features: bool=False,
                  channel_names=None,
                  intensity_functions=[mean_intensity, mean_80_intensity],
                  additional_morphology_functions=[]
                  ): 
    
    
    # get image I sz = (X,Y), dtype = uint8
    check_dtype(I, np.uint8)
    
    # get image I sz = (X,Y), dtype = uint8
    if not isinstance(intensity_image, type(None)):
        check_dtype(intensity_image, np.float32)
    
    #check types of model, either provide a given model or give the name of the model u want to use
    if isinstance(cellpose_net, str):
        model = cellpose.models.CellposeModel(model_type=cellpose_net, gpu=torch.cuda.is_available())

    elif isinstance(cellpose_net, cellpose.models.CellposeModel):
        model = cellpose_net
    else:
        raise ValueError("Enter either a string or a Cellpose Model for cellpose_net")
        
    print(model)
    
    # segment the given image I using a network from cellpose, give cellpose the network u want to use as a parameter cellpose_net = "CPx"
    masks, _, _ = model.eval(I, **eval_kwargs)
    
    # give a flag if you want to correct the masks or not refine = False / True
    if refine:
        
        # give it a threshold that is going to be used for thresholding t = 0.12
        masks = refine_masks(I, masks, t=t)
    
    features = {}
    if extract_morph_features:
            
        features.update(extract_morphology(masks, additional_morphology_functions=additional_morphology_functions))
    
    if not isinstance(intensity_image, type(None)):
        
        out_sz = intensity_image.shape[-2:]

    if not isinstance(out_sz, type(None)):
        if out_sz != I.shape:
            
            masks = cv2.resize(masks, out_sz, interpolation=cv2.INTER_NEAREST)
            I = cv2.resize(I, out_sz, interpolation=cv2.INTER_LINEAR)
    try:
        print(I.shape, intensity_image.shape)
    except:
        pass
        
    # give the function a parameter if you want to extract features (regionprops) and a list of the features you want to extract
    if extract_intensity_features and not isinstance(intensity_image, type(None)):
        
        features.update(extract_intensity(intensity_image, masks, intensity_functions=intensity_functions, channel_names=channel_names))
        
        
    elif extract_intensity_features and isinstance(intensity_image, type(None)):
        
        features.update(extract_intensity(I, masks, intensity_functions=intensity_functions, channel_names=channel_names))

    
    return I, masks, features


def extract_morphology(masks, additional_morphology_functions=None):
    
    props = regionprops_table(masks, properties=PROPERTIES, extra_properties=additional_morphology_functions)
    return props
        

def extract_intensity(I, masks, intensity_functions=None, channel_names=None):
    
    if check_type(intensity_functions, list):
        
        results = {}
        for f in intensity_functions:
            
            results.update(f(I, masks, channel_names=channel_names))
            
    return results


def refine_masks(I, masks, t=0.15):
    
    I = I.copy()/255
    binary_masks = masks.astype(bool)
    binary_masks[I < t] = 0
    refined_masks = binary_masks*masks
    
    # blurring to remove rough edges: b_patch0 = cv2.GaussianBlur(patch0, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)
    refined_masks = cv2.GaussianBlur(refined_masks, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)
    return refined_masks
    

def check_type(inp, target_type):
    if type(inp) != target_type:
        raise TypeError(f"Wrong type for input. Expected {target_type} got {type(inp)}")
    else: 
        return 1


def check_dtype(inp, dtype):
    
    if check_type(inp, np.ndarray):
        if inp.dtype != dtype:
            raise TypeError(f"Wrong dtype for input. Expected {dtype} got {inp.dtype}")
        
          

    