import os


def is_dir(path):
    '''
    args:
        path: path
    return bool
    '''
    path = get_absolute_path(path)
    return os.path.isdir(path)


def is_path(path):
    '''
    args:
        path: path
    return bool
    '''
    path = get_absolute_path(path)
    return os.path.ispath(path)


def get_absolute_path(path):
    '''
    args: 
        path: path
    return: absolute path
    '''
    if path.startswith('~'):
        path = os.path.expanduser(path)
    return os.path.abspath(path)


def get_dir(path):
    path = get_absolute_path(path)
    if is_dir(path):
        return path
    return os.path.split(path)[0]


def mkdir(path):
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path


def make_parent_dir(path):
    parent_dir = get_dir(path)
    mkdir(parent_dir)


def cd(p):
    p = get_absolute_path(p)
    os.chdir(p)


def get_filename(path):
    return os.path.split(path)[1]


def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)


def ls(path='.', suffix=None):
    path = get_absolute_path(path)
    files = os.listdir(path)

    if suffix is None:
        return files

    filtered = []
    for f in files:
        if ends_with(f, suffix, ignore_case=True):
            filtered.append(f)

    return filtered


def ends_with(s, suffix, ignore_case=False):
    """
    suffix: str, list, or tuple
    """
    if type(suffix) == str:
        suffix = [suffix]
    suffix = list(suffix)
    if ignore_case:
        for idx, suf in enumerate(suffix):
            suffix[idx] = str.lower(suf)
        s = str.lower(s)
    suffix = tuple(suffix)
    return s.endswith(suffix)
