import numpy as np
import pandas as pd
import numba as nb
from spectrum_utils.spectrum import MsmsSpectrum
import math
import scipy.sparse as ss
import scipy.sparse.linalg
from typing import Optional
import tensorflow as tf


'''
sepctrum preprocess
discard low intensity, normalize ...
'''

def get_spectrum(mgf, begin, end):
    spectrum = mgf[begin + 16: end]
    mz = [float(i.split("\t")[0].strip()) for i in spectrum]
    intensity = [float(i.split("\t")[1].strip()) for i in spectrum]
    mz = np.array(mz)
    intensity = np.array(intensity)
    return mz, intensity


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def get_num_bins(min_mz: float, max_mz: float, bin_size: float) -> int:
    """
    Compute the number of bins over the given mass range for the given bin
    size.
    """
    return math.ceil((max_mz - min_mz) / bin_size)


def to_vector(spectrum_mz: np.ndarray, spectrum_intensity: np.ndarray,
              min_mz: float = 50.5, bin_size: float = 1.0005079, num_bins: int=2449)\
        -> ss.csr_matrix:
    """
    Convert the given spectrum to a binned sparse SciPy vector.

    """
    bins = ((spectrum_mz - min_mz) / bin_size).astype(np.int32)
    
    # bins = np.append(bins,837)
    # spectrum_intensity = np.append(spectrum_intensity,0.5)
    # print(bins)
    # print(num_bins)
    # print(bins)
    # print(f"spectrum_mz1 = {spectrum_intensity.shape},{np.repeat(0, len(spectrum_intensity)).shape}, bins = {bins.shape}, intensity = {num_bins}")
    vector = ss.csr_matrix(
        (spectrum_intensity, (np.repeat(0, len(spectrum_intensity)), bins)),
        shape=(1, num_bins), dtype=np.float32)
    # print(f"spectrum_mz2 = {spectrum_mz[0], spectrum_mz[-1]}, intensity = {spectrum_intensity[0], spectrum_intensity[-1]}")
    vector = vector.toarray().astype(np.float64)
    vector = np.reshape(vector, (vector.shape[-1]))
    return vector.tolist()
    # return vector / scipy.sparse.linalg.norm(vector)
    

def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used.

    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)
    
# print("hello")
def preprocess(spectrum: MsmsSpectrum,
               mz_min: float = 1.0005079*50.5,
               mz_max: float = 2500,
               min_peaks: int = 10,
               min_mz_range: float = 250,
               remove_precursor_tolerance: Optional[float] = 2,
               min_intensity: float = 0.01,
               max_peaks_used: int = 150,
               scaling: Optional[str] = 'sqrt') -> MsmsSpectrum:
    """
    Preprocess the given spectrum.

    """
    spectrum.is_processed = False
    # print("hello")
    
    if spectrum.is_processed:
        return spectrum

    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    if remove_precursor_tolerance is not None:
        spectrum = spectrum.remove_precursor_peak(
            remove_precursor_tolerance, 'Da')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(min_intensity, max_peaks_used)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks_used)
    # print(spectrum.intensity)
    # spectrum.intensity = _norm_intensity(spectrum.intensity)
    # print(spectrum.intensity)
    inten = _norm_intensity(spectrum.intensity)
    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing.
    spectrum._intensity = inten
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum