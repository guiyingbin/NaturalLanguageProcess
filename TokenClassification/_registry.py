"""
注册模型，数据集，tokenizer，config
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple

__model_labs: Dict[str, Callable[..., Any]] = {}  # 记录模型名称及对应的执行路径
__datasets_labs: Dict[str, Callable[..., Any]] = {}  # 记录数据集
__tokenizer_labs: Dict[str, Callable[..., Any]] = {}  # 记录tokenizer
__config_labs: Dict[str, Callable[..., Any]] = {}  # 记录config


def register_dataset(fn):
    dataset_name = fn.__name__
    __datasets_labs[dataset_name] = fn
    return fn


def register_tokenzier(fn):
    tokenizer_name = fn.__name__
    __tokenizer_labs[tokenizer_name] = fn
    return fn


def register_model(fn):
    model_name = fn.__name__
    __model_labs[model_name] = fn
    return fn


def register_config(fn):
    config_name = fn.__name__
    __config_labs[config_name] = fn
    return fn


def get_config(config_name, *args, **kwargs):
    assert config_name in __config_labs.keys()
    fn = __config_labs[config_name]
    return fn(*args, **kwargs)


def get_dataset(model_name, *args, **kwargs):
    assert model_name in __datasets_labs.keys()
    fn = __datasets_labs[model_name]
    return fn(*args, **kwargs)


def get_tokenizer(tokenizer_name, *args, **kwargs):
    assert tokenizer_name in __tokenizer_labs.keys()
    fn = __tokenizer_labs[tokenizer_name]
    return fn(*args, **kwargs)


def get_model(model_name, *args, **kwargs):
    assert model_name in __model_labs.keys()
    fn = __model_labs[model_name]
    return fn(*args, **kwargs)
