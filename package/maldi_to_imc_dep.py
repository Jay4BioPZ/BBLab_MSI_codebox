import numpy as np
import scipy.ndimage as ndi
import tqdm

def maldi_affine_to_imc(maldi_multi, imc_multi, affine, imc_roi_origin, order=1):
    """
    Transform MALDI images to IMC images using an affine transformation.
    
    Parameters
    ----------
    maldi_multi : np.ndarray
        3D array of MALDI images. [channel, y, x]
    imc_multi : np.ndarray
        3D array of IMC images. [channel, y, x]
    affine : np.ndarray
        3D array of affine transformation manually acquired from napari affinder plugin. Reference have to be mcd IMC images.
    imc_roi_origin : np.ndarray
    order : int
        Interpolation order.
        
    Returns
    -------
    np.ndarray
        3D array of transformed MALDI images, scaled to 0-255.
    """
    print('Transforming MALDI images to register with IMC roi...')
    print('Dimension of MALDI images:', maldi_multi.shape)
    print('Output dimension for each channel:', imc_multi.shape[1:])
    print('Affine transformation:\n', affine)
    print('IMC roi origin:', imc_roi_origin)
    print('Interpolation order:', order)
    print('='*50)
    
    transformed_maldi_multi = np.zeros((maldi_multi.shape[0], imc_multi.shape[1], imc_multi.shape[2]))
    
    for i in range(maldi_multi.shape[0]):
        # get the maldi sample (one channel)
        maldi_sample = maldi_multi[i]
        print('Preprocessing channel ', i)
        # pad the maldi sample to the size of imc roi (global coordinate)
        new_maldi = np.zeros((imc_roi_origin[0]+imc_multi.shape[1], imc_roi_origin[1]+imc_multi.shape[2]))
        new_maldi[:maldi_sample.shape[0], :maldi_sample.shape[1]] = maldi_sample
        # scale the new_maldi to 0-255
        new_maldi = ((new_maldi - np.min(new_maldi)) / (np.max(new_maldi) - np.min(new_maldi)) * 255).astype(np.uint8)
        
        # transform the maldi sample to the imc roi
        print('Transforming...')
        transformed_maldi = ndi.affine_transform(new_maldi, np.linalg.inv(affine), order = order)
        transformed_maldi_multi[i] = transformed_maldi[imc_roi_origin[0]:, imc_roi_origin[1]:]
        print('-'*50)
    return transformed_maldi_multi