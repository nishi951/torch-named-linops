from types import SimpleNamespace

from torch import Tensor

from torchlinops.linops.pad_last import PadLast, crop_slice_from_pad, pad_to_size

__all__ = ["center_pad", "center_crop"]


def center_pad(x: Tensor, im_size: tuple[int, ...], pad_im_size: tuple[int, ...]):
    """
    Center pads the input tensor to the specified size.

    Parameters
    ----------
    x : Tensor
        The input tensor to be padded.
    im_size : tuple of int
        The original size of the input tensor.
    pad_im_size : tuple of int
        The desired size after padding.

    Returns
    -------
    Tensor
        The padded tensor.

    Notes
    -----
    This function uses the `PadLast` operator to perform the padding operation.
    It calculates the necessary padding dimensions and applies the padding
    to ensure that the output tensor matches the `pad_im_size` while preserving
    the center of the original tensor.
    """
    ndim = len(im_size)
    pad = pad_to_size(im_size, pad_im_size)
    crop_slice = crop_slice_from_pad(pad)
    pad_linop = SimpleNamespace(
        im_size=im_size,
        pad_im_size=pad_im_size,
        D=ndim,
        pad=pad,
        crop_slice=crop_slice,
    )
    return PadLast.fn(pad_linop, x)


def center_crop(x: Tensor, im_size: tuple[int, ...], crop_im_size: tuple[int, ...]):
    """
    Center crops the input tensor to the specified size.

    Parameters
    ----------
    x : Tensor
        The input tensor to be cropped.
    im_size : tuple of int
        The original size of the input tensor.
    crop_im_size : tuple of int
        The desired size after cropping.

    Returns
    -------
    Tensor
        The cropped tensor.

    Notes
    -----
    This function uses the `PadLast` operator to perform the cropping operation.
    It calculates the necessary cropping dimensions and applies the cropping
    from the center of the original tensor to achieve the specified `crop_im_size`.
    """
    ndim = len(im_size)
    pad = pad_to_size(crop_im_size, im_size)
    crop_slice = crop_slice_from_pad(pad)
    crop_linop = SimpleNamespace(
        im_size=crop_im_size,
        pad_im_size=im_size,
        D=ndim,
        pad=pad,
        crop_slice=crop_slice,
    )
    return PadLast.adj_fn(crop_linop, x)
