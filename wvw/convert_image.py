"""
MIT No Attribution

Copyright 2023 Alexander Auras

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import numpy.typing
from PySide6.QtGui import QImage


def numpy_to_qimg(
    npimg: numpy.typing.NDArray[np.float32 | np.float64 | np.uint8 | np.ubyte],
) -> QImage:
    """
    Convert a numpy array into a QImage.

    :param npimg: A numpy array of type `np.float_` with values in [0.0, 1.0] or of type `np.uint8`.
                  The shape must match (..., channels, height, width), channels should be 3 (RGB) or 4 (RGBA).
    :return: A QImage with format `QImage.Format.Format_ARGB32` with the same shape and content as `npimg`.
    """  # noqa: E501
    if not isinstance(npimg, np.ndarray):  # pyright: ignore [reportUnnecessaryIsInstance]
        raise ValueError(f"Expected a numpy array, got {type(npimg).__name__}")
    if not np.issubdtype(npimg.dtype, np.floating) and npimg.dtype not in [np.uint8, np.ubyte]:
        raise ValueError(
            f"Unsupported array dtype (expected floating type or np.uint8/np.ubyte, got {npimg.dtype})"  # noqa: E501
        )
    if npimg.ndim < 3:
        raise ValueError(f"Missing dimensions in numpy array (expected 3, got {npimg.ndim})")
    for _ in range(npimg.ndim - 3):
        npimg = npimg[0]
    if npimg.shape[0] != 4:
        if npimg.shape[0] == 3:
            npimg = np.concatenate((npimg, np.ones_like(npimg[0:1])), axis=0)
        else:
            raise ValueError(f"Invalid number of channels (expected 3 or 4, got {npimg.shape[-3]})")
    if npimg.shape[1] == 0 or npimg.shape[2] == 0:
        return QImage(bytes(), 0, 0, 0, QImage.Format.Format_ARGB32)
    if npimg.size >= 2147483647:
        raise ValueError(f"Array is to large (max supported size is 2147483647, got {npimg.size})")
    if np.issubdtype(npimg.dtype, np.floating):
        npimg = npimg.astype(np.float32)
    elif np.issubdtype(npimg.dtype, np.unsignedinteger):
        npimg = npimg.astype(np.float32) / np.iinfo(npimg.dtype).max  # pyright: ignore [reportCallIssue, reportArgumentType]
    else:
        raise ValueError(
            f"Unsupported array dtype (expected floating or unsigned integer type, got {npimg.dtype})"  # noqa: E501
        )
    if npimg.min() < 0.0 or npimg.max() > 1.0:
        raise ValueError(
            f"Array value outside of valid range (must be in [0.0, 1.0], got [{npimg.min()}, {npimg.max()}])"  # noqa: E501
        )
    bytes_ = np.zeros((npimg.size,))
    bytes_[0::4] = npimg[2].flatten() * 255.0
    bytes_[1::4] = npimg[1].flatten() * 255.0
    bytes_[2::4] = npimg[0].flatten() * 255.0
    bytes_[3::4] = npimg[3].flatten() * 255.0
    return QImage(
        np.ascontiguousarray(bytes_.astype(np.uint8)).data,
        npimg.shape[2],
        npimg.shape[1],
        npimg.shape[2] * npimg.shape[0],
        QImage.Format.Format_ARGB32,
    )


def qimg_to_numpy(qimg: QImage) -> numpy.typing.NDArray[np.float32]:
    """
    Convert a QImage into a numpy array.

    :param qimg: A QImage, the format must be convertable to `QImage.Format.Format_ARGB32`.
    :return: A numpy array of type `np.float32` with values in [0.0, 1.0] and shape (4, height, width).
    Size and content equal the shape and content of `qimg` in RGBA-order.
    """  # noqa: E501
    if not isinstance(qimg, QImage):  # pyright: ignore [reportUnnecessaryIsInstance]
        raise ValueError(f"Expected a QImage, got {type(qimg)}")
    if qimg.width() == 0 or qimg.height() == 0:
        return np.zeros((0,), dtype=np.float32)
    if qimg.format() != QImage.Format.Format_ARGB32:
        qimg = qimg.convertToFormat(QImage.Format.Format_ARGB32)
    bytes_ = np.frombuffer(qimg.bits(), dtype=np.uint8)
    npimg = np.zeros((4, qimg.height(), qimg.width()), dtype=np.uint8)
    npimg[0] = bytes_[2::4].reshape(qimg.height(), qimg.width())
    npimg[1] = bytes_[1::4].reshape(qimg.height(), qimg.width())
    npimg[2] = bytes_[0::4].reshape(qimg.height(), qimg.width())
    npimg[3] = bytes_[3::4].reshape(qimg.height(), qimg.width())
    return npimg.astype(np.float32) / 255.0
