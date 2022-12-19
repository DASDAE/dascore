"""
A script to create an index of the structure of the DASCore package.
"""
from __future__ import annotations

import inspect
import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from types import MethodType, ModuleType, FunctionType
from typing import Literal


def _unwrap_obj(obj):
    """Unwrap a decorated object."""
    while getattr(obj, "__wrapped__", None) is not None:
        obj = obj.__wrapped__
    return obj


def _get_file_path(obj):
    """Try to get the file of a python object."""
    obj = _unwrap_obj(obj)
    try:
        path = inspect.getfile(obj)
    except TypeError:
        path = ""
    return Path(path)


def _get_base_address(path, base_path):
    """
    Get the base address inherent in the path.

    For example, dascore/core/patch.py should return

    dascore.core.patch
    """
    try:
        out = Path(path).relative_to(Path(base_path))
    except ValueError:
        return ""
    new = str(out).replace("/__init__.py", "").replace(".py", "")
    return new.replace("/", ".")


def parse_project(obj, key=None):
    """Parse the project create dict of data and data_type"""

    def yield_get_submodules(obj, base_path):
        """Dynamically load submodules that may not have been imported."""
        path = Path(inspect.getfile(obj))
        if not isinstance(obj, ModuleType) or path.name != "__init__.py":
            return
        submodules = path.parent.glob("*")
        for submod_path in submodules:
            is_dir = submod_path.is_dir()
            is_init = submod_path.name.endswith("__init__.py")
            # this is a directory, look for corresponding __init__.py
            if is_dir and (submod_path / "__init__.py").exists():
                mod_name = str(submod_path.relative_to(base_path)).replace(os.sep, ".")
                mod = import_module(mod_name)
                yield mod_name, mod
            elif submod_path.name.endswith(".py") and not is_init:
                mod_name = (
                    str(submod_path.relative_to(base_path))
                    .replace(".py", "")
                    .replace(os.sep, ".")
                )
                mod = import_module(mod_name)
                yield mod_name, mod

    def get_type(
        obj, parent_is_class=False
    ) -> None | Literal["module", "function", "method", "class"]:
        """Return a string of the type of object."""
        obj = _unwrap_obj(obj)
        if isinstance(obj, ModuleType):
            return "module"
        elif isinstance(obj, MethodType):
            return "method"
        elif isinstance(obj, FunctionType):
            # since this is a class not instance we need this check.
            if parent_is_class:
                return "method"
            else:
                return "function"
        elif isinstance(obj, type):
            return "class"
        return None

    def extract_data(obj, parent_is_class):
        """
        Make lists of attributes, methods, etc.

        These can be feed to render templates.
        """

        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            sig = None

        docstr = inspect.getdoc(obj)
        dtype = get_type(obj, parent_is_class)
        data = defaultdict(list)
        data["docstring"] = docstr
        data["signature"] = sig
        data["data_type"] = dtype
        data["short_description"] = docstr.split("\n")[0]
        data["object"] = obj
        # get sub-motdules, methods, functions, etc. (just one level deep)

        subs = list(inspect.getmembers(obj)) + list(
            yield_get_submodules(obj, base_path)
        )
        for name, sub_obj in subs:
            if name.startswith("_"):
                continue
            sub_obj = _unwrap_obj(sub_obj)
            sub_dtype = get_type(sub_obj, dtype == "class")
            # for modules, skip entities that aren't children
            if dtype == 'module':
                path, sub_path = _get_file_path(obj), _get_file_path(sub_obj)
                if not str(path).replace('/__init__.py', '') in str(sub_path):
                    continue
            data[sub_dtype].append(str(id(sub_obj)))

        return data

    def get_data(obj, key, base_path, parent_is_class):
        """Get data from object."""
        path = inspect.getfile(obj)
        base_address = _get_base_address(path, base_path)
        data = extract_data(obj, parent_is_class)
        data["base_path"] = Path(base_path)
        data["path"] = Path(path)
        data["key"] = key
        data["name"] = key.split(".")[-1]
        data["base_address"] = base_address
        return data

    def get_address(obj, rel_path):
        """Get the address for an object."""
        base_list = (
            str(rel_path)
            .replace(f"{os.sep}__init__.py", "")
            .replace(".py", "")
            .split(os.sep)
        )
        if not isinstance(obj, ModuleType) and hasattr(obj, "__name__"):
            name_list = [obj.__name__]
        else:
            name_list = []
        return ".".join(base_list + name_list)

    def traverse(obj, data_dict, base_path, key=None, parent_is_class=False):
        """Traverse tree, populate data_dict"""
        obj = _unwrap_obj(obj)
        obj_id = str(id(obj))
        path = _get_file_path(obj)
        # this is something outside of dascore
        if str(base_path) not in str(path) or obj_id in data_dict:
            return
        # load all the modules first
        if isinstance(obj, ModuleType):
            key = get_address(obj, path.relative_to(base_path))
            data_dict[obj_id] = get_data(obj, key, base_path, parent_is_class)
            for _, mod in yield_get_submodules(obj, base_path):
                traverse(mod, data_dict, base_path)
            for name, obj in inspect.getmembers(obj):
                # recurse non-private methods
                if not name.startswith("_"):
                    traverse(obj, data_dict, base_path, f"{key}.{name}", False)
        # then handle non-modules
        else:
            data_dict[str(id(obj))] = get_data(obj, key, base_path, parent_is_class)
            # recurse attributes and methods of classes
            if inspect.isclass(obj):
                for sub_name, sub_obj in inspect.getmembers(obj):
                    if sub_name.startswith("_") or not callable(sub_obj):
                        continue
                    sub_path = _get_file_path(sub_obj)
                    if not str(base_path) in str(sub_path):
                        continue
                    sub_key = f"{key}.{sub_name}"
                    # make sure this is where the method is defined else skip
                    if sub_path != path:
                        continue
                    traverse(sub_obj, data_dict, base_path, sub_key, True)

    # key = key or getattr(obj, "__name__", None)
    base_path = Path(_get_file_path(obj)).parent.parent
    data_dict = {}
    traverse(obj, data_dict, base_path)
    # traverse(obj, key, data_dict, str(base_path), parent_path=base_path)
    return data_dict


def get_alias_mapping(module, key=None):
    """Return a dict of {object_path: id} to construct cross refs."""

    def traverse_simple(obj, key, data_dict, base_path):
        """Traverse the tree and write out markdown."""
        obj = _unwrap_obj(obj)
        obj_id = str(id(obj))
        path = _get_file_path(obj)
        if not _get_base_address(path, base_path):
            return
        data_dict[key] = obj_id
        for (member_name, member) in inspect.getmembers(obj):
            if member_name.startswith("_"):
                continue
            new_key = ".".join([key, member_name])
            if new_key.split(".")[-1] in key:
                continue
            traverse_simple(member, new_key, data_dict, base_path)

    data_dict = {}
    key = key or getattr(module, "__name__", None)
    base_path = Path(_get_file_path(module)).parent.parent
    traverse_simple(module, key, data_dict, base_path)
    return data_dict
