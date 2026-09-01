##############################################################
# Where specsim's own data and config files live
###############################################################
#
# Every path shipped in a config file or defaulted in code -- './data/...',
# './configs/instruments/...' -- is written relative to the specsim source
# tree, not to wherever the user happens to be running from. Resolving them
# against SPECSIM_ROOT rather than the current working directory is what lets
# a run work from any folder, so a user only has to write their own .cfg.
#
# This module imports nothing from specsim, so anything (config, plot, the
# detector modules) can depend on it without an import cycle.

import os

# <specsim root>/specsim/paths.py -> <specsim root>
SPECSIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(SPECSIM_ROOT, 'data') + os.sep
BUNDLED_CONFIGS = os.path.join(SPECSIM_ROOT, 'configs')


def from_root(*parts):
    """
    Absolute path to something inside the specsim tree.

    inputs
    ------
    *parts : str
        path components below the specsim root, e.g.
        from_root('data', 'filters') -> '<specsim root>/data/filters'

    output
    ------
    str, absolute path
    """
    return os.path.join(SPECSIM_ROOT, *parts)


def resolve(path, base=SPECSIM_ROOT):
    """
    Resolve a possibly-relative config path against the specsim tree,
    leaving absolute paths (a user pointing at their own data) alone.

    inputs
    ------
    path : str
        path as written in a .cfg or instrument YAML
    base : str
        directory relative paths are taken against; defaults to the
        specsim root

    output
    ------
    str, absolute path
    """
    resolved = path if os.path.isabs(path) else os.path.join(base, path)
    # normpath tidies the './' that config paths are written with, but drops a
    # trailing separator -- which some callers rely on, since they build
    # filenames by string concatenation (e.g. sonora_folder + 'sp_t...').
    normalized = os.path.normpath(resolved)
    return normalized + os.sep if resolved.endswith(('/', os.sep)) else normalized
