"""
Spectrogram augmentation utilities.

This module provides functions for spectrogram augmentation using techniques
like time masking and frequency masking as described in SpecAugment paper.
"""

import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


def apply_time_mask(
    spectrogram: np.ndarray, time_param: int = 10, num_masks: int = 1
) -> np.ndarray:
    """
    Apply time masking to a spectrogram for augmentation.

    Args:
        spectrogram: Input spectrogram with shape [time_steps, features]
        time_param: Maximum length of time mask
        num_masks: Number of masks to apply

    Returns:
        Augmented spectrogram
    """
    spec_tf = tf.convert_to_tensor(spectrogram, dtype=tf.float32)
    masked_spec = spec_tf

    for _ in range(num_masks):
        masked_spec = tfio.audio.time_mask(masked_spec, param=time_param)

    return masked_spec.numpy()


def apply_freq_mask(
    spectrogram: np.ndarray, freq_param: int = 10, num_masks: int = 1
) -> np.ndarray:
    """
    Apply frequency masking to a spectrogram for augmentation.

    Args:
        spectrogram: Input spectrogram with shape [time_steps, features]
        freq_param: Maximum number of frequency channels to mask
        num_masks: Number of masks to apply

    Returns:
        Augmented spectrogram
    """
    spec_tf = tf.convert_to_tensor(spectrogram, dtype=tf.float32)
    masked_spec = spec_tf

    for _ in range(num_masks):
        masked_spec = tfio.audio.freq_mask(masked_spec, param=freq_param)

    return masked_spec.numpy()


def augment_spectrogram(
    spectrogram: np.ndarray,
    time_param: int = 10,
    time_masks: int = 1,
    freq_param: int = 10,
    freq_masks: int = 1,
) -> np.ndarray:
    """
    Apply both time and frequency masking to a spectrogram for augmentation.

    Args:
        spectrogram: Input spectrogram with shape [time_steps, features]
        time_param: Maximum length of time mask
        time_masks: Number of time masks to apply
        freq_param: Maximum number of frequency channels to mask
        freq_masks: Number of frequency masks to apply

    Returns:
        Augmented spectrogram
    """
    # First apply time masking
    if time_masks > 0:
        spectrogram = apply_time_mask(spectrogram, time_param, time_masks)

    # Then apply frequency masking
    if freq_masks > 0:
        spectrogram = apply_freq_mask(spectrogram, freq_param, freq_masks)

    return spectrogram


def augment_data(
    data_list: List[np.ndarray],
    time_param: int = 10,
    time_masks: int = 1,
    freq_param: int = 10,
    freq_masks: int = 1,
    augmentation_factor: int = 1,
) -> List[np.ndarray]:
    """
    Augment a list of spectrograms using time and frequency masking.

    Args:
        data_list: List of spectrograms to augment
        time_param: Maximum length of time mask
        time_masks: Number of time masks to apply
        freq_param: Maximum number of frequency channels to mask
        freq_masks: Number of frequency masks to apply
        augmentation_factor: Number of augmented copies to create for each original spectrogram

    Returns:
        List of original and augmented spectrograms
    """
    if augmentation_factor <= 0:
        return data_list

    augmented_list = list(data_list)  # Start with original data

    for _ in range(augmentation_factor):
        for spec in data_list:
            augmented = augment_spectrogram(
                spec, time_param, time_masks, freq_param, freq_masks
            )
            augmented_list.append(augmented)

    return augmented_list


def random_augment(
    spectrogram: np.ndarray,
    max_time_param: int = 20,
    max_time_masks: int = 3,
    max_freq_param: int = 15,
    max_freq_masks: int = 3,
) -> np.ndarray:
    """
    Apply random augmentation to a spectrogram.

    This function applies a random number of time and frequency masks with
    random parameters within specified bounds.

    Args:
        spectrogram: Input spectrogram with shape [time_steps, features]
        max_time_param: Maximum length of time mask
        max_time_masks: Maximum number of time masks to apply
        max_freq_param: Maximum number of frequency channels to mask
        max_freq_masks: Maximum number of frequency masks to apply

    Returns:
        Augmented spectrogram
    """
    # Generate random parameters
    time_param = random.randint(1, max_time_param)
    time_masks = random.randint(1, max_time_masks)
    freq_param = random.randint(1, max_freq_param)
    freq_masks = random.randint(1, max_freq_masks)

    # Apply augmentation
    return augment_spectrogram(
        spectrogram,
        time_param=time_param,
        time_masks=time_masks,
        freq_param=freq_param,
        freq_masks=freq_masks,
    )
