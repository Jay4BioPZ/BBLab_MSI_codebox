import numpy as np
import scipy.ndimage as ndi
from joblib import Parallel, delayed

def affine_translate(dx, dy):
    # create the translation matrix 3x3
    translation_matrix = np.array([[1, 0, dx],
                                    [0, 1, dy],
                                    [0, 0, 1]])
    return translation_matrix

def transform_single_channel_eff(maldi_sample, affine, imc_roi_origin, output_shape, order):
    # perform transformation using the global coordinates in napari (padding zeros to the empty space)
    maldi_norm = ((maldi_sample - np.min(maldi_sample)) / (np.max(maldi_sample) - np.min(maldi_sample)) * 255).astype(np.uint8)
    
    # Apply affine transformation
    transformed = ndi.affine_transform(maldi_norm, np.linalg.inv(affine) @ affine_translate(imc_roi_origin[0], imc_roi_origin[1]), order=order, output_shape=output_shape)
    return transformed

def transform_single_channel(maldi_sample, affine, imc_roi_origin, output_shape, order):
    # perform transformation using additional translation, avoid zero padding and big-image operation (much faster)
    """Apply affine transformation to a single MALDI channel and crop to IMC region."""
    # Normalize to 0-255
    maldi_pad = np.zeros((imc_roi_origin[0]+output_shape[0], imc_roi_origin[1]+output_shape[1]))
    maldi_pad[:maldi_sample.shape[0], :maldi_sample.shape[1]] = maldi_sample
    maldi_pad = ((maldi_pad - np.min(maldi_pad)) / (np.max(maldi_pad) - np.min(maldi_pad)) * 255).astype(np.uint8)
    
    # Apply affine transformation
    transformed = ndi.affine_transform(maldi_pad, np.linalg.inv(affine), order=order)
    
    # Crop to IMC ROI
    return transformed[imc_roi_origin[0]:imc_roi_origin[0]+output_shape[0], 
                        imc_roi_origin[1]:imc_roi_origin[1]+output_shape[1]]



def maldi_affine_to_imc(maldi_multi, imc_multi, affine, imc_roi_origin, order=1, n_jobs=-1, fast_alg=False):
    """
    Optimized MALDI to IMC transformation using parallelization.
    
    Parameters
    ----------
    maldi_multi : np.ndarray
        3D array of MALDI images. [channel, y, x]
    imc_multi : np.ndarray
        3D array of IMC images. [channel, y, x]
    affine : np.ndarray
        3D affine transformation matrix from napari affinder plugin.
    imc_roi_origin : tuple
        Origin (y, x) of IMC ROI in global coordinates.
    order : int, optional
        Interpolation order, default is 1 (bilinear).
    n_jobs : int, optional
        Number of parallel jobs (-1 uses all available cores).
        
    Returns
    -------
    np.ndarray
        3D array of transformed MALDI images, scaled to 0-255.
    """
    output_shape = imc_multi.shape[1:]  # (height, width)
    
    # Parallelize transformation for each channel
    # Use fast algorithm if specified
    if fast_alg:
        transformed_maldi_multi = Parallel(n_jobs=n_jobs)(
            delayed(transform_single_channel_eff)(maldi_multi[i], affine, imc_roi_origin, output_shape, order)
            for i in range(maldi_multi.shape[0])
        )
    else:
        transformed_maldi_multi = Parallel(n_jobs=n_jobs)(
            delayed(transform_single_channel)(maldi_multi[i], affine, imc_roi_origin, output_shape, order)
            for i in range(maldi_multi.shape[0])
        )
    # Stack transformed channels
    transformed_maldi_multi = np.stack(transformed_maldi_multi, axis=0)
    return transformed_maldi_multi