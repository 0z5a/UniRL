import importlib.metadata
from importlib.metadata import PathDistribution
import importlib.util
from functools import partial
from pathlib import Path

from diffusers.utils.import_utils import compare_versions
from loguru import logger
from packaging.version import parse

_index_kits_available = importlib.util.find_spec("index_kits") is not None
try:
    import index_kits as _index_kits
    _index_kits_version = _index_kits.__version__
    _spec = importlib.util.find_spec("index_kits")
    if _spec is not None:
        _locations = list(_spec.submodule_search_locations or [])
        _origin = _spec.origin or (_locations[0] if _locations else "")
        _index_kits_source = "submodule: deps/IndexKits" if "deps/IndexKits" in _origin else "system"
    else:
        _index_kits_source = "unknown"
    logger.debug(f"Successfully imported index-kits version {_index_kits_version} ({_index_kits_source})")
except importlib.metadata.PackageNotFoundError:
    _index_kits_version = False


def is_package_version(package: str, operation: str, version: str):
    """
    Compares the current Accelerate version to a given reference with an operation.

    Args:
        package (str): The package name to check.
        operation (str): A string representation of an operator, such as `">"` or `"<="`
        version (str): A version string
    """
    package_spec = importlib.util.find_spec(package.replace("-", "_"))
    if package_spec is None:
        return False

    get_version_from_metadata = False
    try:
        distribution = importlib.metadata.distribution(package)

        # distribution path is consistent with find_spec
        if Path(package_spec.origin).parent.parent == distribution._path.parent:
            get_version_from_metadata = True
    except:
        get_version_from_metadata = False

    if get_version_from_metadata:
        package_version = importlib.metadata.version(package)
    else:
        try:
            module = importlib.import_module(package.replace("-", "_"))
            package_version = getattr(module, "__version__", None)
            if package_version is None:
                return False
        except:
            return False

    return compare_versions(parse(package_version), operation, version)


def require_version(package: str, version: str, src: str):
    if not is_package_version(package, ">=", version):
        raise ImportError(f"{src} requires {package}>={version}, please install or upgrade {package}.")


is_index_kits_version = partial(is_package_version, "index_kits")
