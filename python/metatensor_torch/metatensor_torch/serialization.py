import os
import pathlib
import warnings
from typing import BinaryIO, Union

import torch


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0":
    from .documentation import (
        Labels,
        LabelsEntry,
        TensorBlock,
        TensorMap,
    )
else:
    Labels = torch.classes.metatensor.Labels
    LabelsEntry = torch.classes.metatensor.LabelsEntry
    TensorBlock = torch.classes.metatensor.TensorBlock
    TensorMap = torch.classes.metatensor.TensorMap


# the functions in this file are written in a relatively strange way to enable them
# taking `Path` and `BinaryIO` arguments when running in Python mode, while they only
# support str in TorchScript mode


def load(file: str) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given ``file``.

    :py:class:`TensorMap` are serialized using the ``.mts`` format, i.e. a ZIP file
    without compression (storage method is ``STORED``), where each file is stored as a
    ``.npy`` array. See the C API documentation for more information on the format.

    :param file: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load(file=file)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load(file=file)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load(file=str(file.resolve()))
        else:
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_buffer(buffer=buffer)


# modify the annotations in a way such that the TorchScript compiler does not see these,
# but sphinx does for documentation.
load.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_block(file: str) -> TensorBlock:
    """
    Load previously saved :py:class:`TensorBlock` from the given ``file``.

    :param path: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_block(file=file)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_block(file=file)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_block(file=str(file.resolve()))
        else:
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_block_buffer(buffer=buffer)


load_block.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_labels(file: str) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given ``file``.

    :param file: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_labels(file=file)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_labels(file=file)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_labels(file=str(file.resolve()))
        else:
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_labels_buffer(buffer=buffer)


load_labels.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_mmap(file: str) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given ``file`` using
    memory-mapped I/O. Arrays are created internally as read-only mmap-backed arrays.

    :param file: path of the file to load.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_mmap(file=file)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_mmap(file=file)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_mmap(file=str(file.resolve()))
        else:
            raise TypeError(
                "load_mmap only supports file paths (str or Path), "
                "not file-like objects"
            )


load_mmap.__annotations__["file"] = Union[str, pathlib.Path]


_EMPTY_LABELS = Labels(names=["_"], values=torch.zeros(0, 1, dtype=torch.int32))


def load_mmap_partial(
    file: str,
    keys: Labels = _EMPTY_LABELS,
    samples: Labels = _EMPTY_LABELS,
    properties: Labels = _EMPTY_LABELS,
) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given ``file`` using
    memory-mapped I/O, selecting only a subset of data.

    Each :py:class:`Labels` parameter acts as a filter:

    - an empty Labels (zero entries, the default) selects all entries along
      that axis
    - otherwise, only matching entries are included

    :param file: path of the file to load.
    :param keys: selection for which blocks to load.
    :param samples: selection for which sample rows to keep.
    :param properties: selection for which property columns to keep.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments
            for ``file`` are supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_mmap_partial(
            file=file, keys=keys, samples=samples, properties=properties
        )
    else:
        if isinstance(file, str):
            path = file
        elif isinstance(file, pathlib.Path):
            path = str(file.resolve())
        else:
            raise TypeError(
                "load_mmap_partial only supports file paths (str or Path), "
                "not file-like objects"
            )
        return torch.ops.metatensor.load_mmap_partial(
            file=path, keys=keys, samples=samples, properties=properties
        )


load_mmap_partial.__annotations__["file"] = Union[str, pathlib.Path]


def load_block_mmap(file: str) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from the given ``file`` using
    memory-mapped I/O. Arrays are created internally as read-only mmap-backed arrays.

    :param file: path of the file to load.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_block_mmap(file=file)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_block_mmap(file=file)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_block_mmap(file=str(file.resolve()))
        else:
            raise TypeError(
                "load_block_mmap only supports file paths (str or Path), "
                "not file-like objects"
            )


load_block_mmap.__annotations__["file"] = Union[str, pathlib.Path]


def save(file: str, data: Union[TensorMap, TensorBlock, Labels]):
    """
    Save the given data (either :py:class:`TensorMap`, :py:class:`TensorBlock`, or
    :py:class:`Labels`) to the given ``file``.

    If the file already exists, it is overwritten. The recomended file extension when
    saving data is ``.mts``, to prevent confusion with generic ``.npz`` files.

    :param file: path of the file where to save the data, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.

    :param data: data to serialize and save
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.save(file=file, data=data)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.save(file=file, data=data)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.save(file=str(file.resolve()), data=data)
        else:
            # assume a file-like object
            buffer = torch.ops.metatensor.save_buffer(data=data)
            assert isinstance(buffer, torch.Tensor)
            file.write(buffer.numpy().tobytes())


save.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]
