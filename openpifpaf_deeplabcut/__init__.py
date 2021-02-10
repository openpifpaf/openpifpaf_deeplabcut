
import openpifpaf

from .datamodule import DeepLabCut

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def register():
    openpifpaf.DATAMODULES['deeplabcut'] = DeepLabCut
