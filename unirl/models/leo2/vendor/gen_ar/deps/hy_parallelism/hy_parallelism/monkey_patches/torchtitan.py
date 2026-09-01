
import sys
import types
import importlib
import importlib.util
from packaging import version

import torch

def _patch_missing_module(parent_module, module_path: str):
    """Create missing module path for compatibility."""
    parts = module_path.split('.')
    current = parent_module
    parent_path = parent_module.__name__ if hasattr(parent_module, '__name__') else ''
    
    for part in parts:
        if not hasattr(current, part):
            module = types.ModuleType(part)
            setattr(current, part, module)
            full_path = f"{parent_path}.{part}" if parent_path else part
            sys.modules[full_path] = module
        current = getattr(current, part)
        parent_path = f"{parent_path}.{part}" if parent_path else part


def _try_import_or_patch(module_name: str, parent_module, module_path: str):
    """Try to import a module, patch if import fails."""
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        _patch_missing_module(parent_module, module_path)


# for torchtitan compatibility
def _apply_compatibility_patches():
    """Apply compatibility patches for missing modules."""
    # Skip import for compatibility with older PyTorch versions
    sys.modules['torchtitan.components.quantization'] = types.ModuleType('quantization')


def _is_torchtitan_installed():
    """Check if torchtitan is installed without importing it."""
    return importlib.util.find_spec('torchtitan') is not None


def patch_torchtitan_quantization():
    # requires torch attention varlen
    _apply_compatibility_patches()
    if not _is_torchtitan_installed():
        raise RuntimeError('Torchtitan is not installed. Please install it with: pip install torchtitan')