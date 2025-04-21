# -*- coding: utf-8 -*-



from typing import List, Tuple, Dict, Callable, Union, Any
import sys
import os
import numpy as np
import json

__all__ = [
    'mkdir',
    'mkdirs',
    'load_json',
    'write_json',
    'loc_key_via_value'
]


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(json_path: str):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data: Dict, json_path: str) -> None:
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)


def loc_key_via_value(data: Dict, value_to_find):
    '''
    :param value_to_find: arbitary value to find
    :return:
    '''
    keys = [key for key, value in data.items() if value == value_to_find]
    return keys
