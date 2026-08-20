"""
The immutable, fingerprintable unit of work DASCore's workflows are built of.

A [`Task`](`dascore.workflow.task.Task`) is a frozen pydantic model whose
fields are the parameters of one operation. Because the parameters are the
object, a task can be compared, hashed, written to a file and read back, and
identified by a fingerprint: a digest of which task it is, which version of
it, and what it was given.

Examples
--------
>>> from dascore.workflow import Task
>>>
>>> class AddNumberExample(Task):
...     '''Add a number to what it is given.'''
...     value: int = 1
...
...     def run(self, number):
...         return number + self.value
>>>
>>> task = AddNumberExample(value=2)
>>> assert task.run(1) == 3
>>> # The same task, however it was built, has the same fingerprint.
>>> assert task.fingerprint == AddNumberExample(value=2).fingerprint
>>> assert task.fingerprint != AddNumberExample(value=3).fingerprint
"""

from __future__ import annotations

import inspect
import warnings
import weakref
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import ConfigDict, model_validator
from pydantic.fields import FieldInfo

from dascore.compat import array, is_array_like
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.registry import (
    NAMESPACE_SEP,
    TAG_FIELD,
    get_model_tag,
    resolve_model_tag,
    resolve_tagged_model,
)
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow.serialize import (
    DOCUMENT,
    decode,
    digest,
    encode,
    model_values,
    read_workflow,
    write_workflow,
)

# The keys a task's document holds beside its parameters.
_VERSION_KEY = "version"
_PARAMS_KEY = "params"


class Task(DascoreBaseModel):
    """
    Base class for a fingerprintable, serializable operation.

    Subclasses declare their parameters as fields and implement `run`. Every
    instance is frozen, so a task can be shared, cached and reused; changing
    one means making another with `update`.

    What `run` takes is what the task is given when it runs, positionally
    and in the order its parameters are declared; everything the task was
    *configured* with is a field. Inputs go by position rather than by
    name because the order they are given in is part of what the task did.

    An array given as a parameter is marked read-only in place, as a
    patch's data is, so that the fingerprint cannot come to describe values
    the task no longer holds. Nothing is copied, so the array marked is the
    caller's own: pass a copy of a buffer which is still being written to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _own_the_arrays(cls, data: Any) -> Any:
        """Take ownership of any array a task was handed."""
        return own_arrays(data) if isinstance(data, dict) else data

    # Bumped by a subclass whenever the same parameters should mean a
    # different answer, so that old fingerprints do not name the new
    # behaviour. It is a ClassVar rather than a field: the version belongs to
    # the class, not to the call.
    __version__: ClassVar[str] = "1.0"

    @property
    def tag(self) -> str | None:
        """
        Return the registered name of this task's class.

        None means the class cannot be named in a document, which a
        parametrized generic cannot; see `dascore.models.registry`.
        """
        return get_model_tag(type(self))

    @cached_property
    def fingerprint(self) -> str:
        """
        Return the digest which identifies this task and its parameters.

        The fingerprint names the top level package, the class name, the
        version and the parameters. It deliberately leaves out the submodule
        the class lives in, the source of its body, and the backend which
        will run it: DASCore moves its own code around, and neither a move
        within a package nor a faster kernel makes an operation a different
        operation.
        """
        return self.fingerprint_at(self.__version__)

    def fingerprint_at(self, version: str) -> str:
        """
        Return the fingerprint this task has at a stated version.

        For reading a document back: the version is part of what a task is,
        so a document written before its class moved on records a
        fingerprint this task no longer has. Asking for it at the version
        the document names says what it was, which is how an edited file
        can still be told from an old one.

        Parameters
        ----------
        version
            The version to compute it at, as `__version__` spells one.
        """
        # The parameters go in as the objects they are: `digest` encodes
        # what it is given, and encoding them here as well would put every
        # tagged value -- an array, a time, a quantity -- through the
        # escape which exists for a mapping that spells a tag itself.
        payload = {
            "task": _qualified_tag(type(self)),
            _VERSION_KEY: version,
            _PARAMS_KEY: self._params(),
        }
        return digest(payload)

    def run(self, *args, **kwargs) -> Any:
        """Execute the task. Subclasses must implement this."""
        msg = f"{type(self).__name__} does not implement run."
        raise NotImplementedError(msg)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Run the task; see `run`.

        Defined rather than aliased so that a subclass's own `run` is the
        one called, and so a task can stand anywhere a function of its
        inputs can -- `spool.map(task)`, most of all.
        """
        return self.run(*args, **kwargs)

    def update(self, **kwargs) -> Self:
        """
        Return a new task with some parameters changed.

        Examples
        --------
        >>> from dascore.workflow import Task
        >>> class ScaleExample(Task):
        ...     factor: float = 1.0
        >>> assert ScaleExample().update(factor=2.0).factor == 2.0
        """
        return type(self)(**{**self._params(), **kwargs})

    def to_dict(self) -> dict[str, Any]:
        """
        Return a document which describes this task.

        The class is named by its registered tag and by nothing else: the
        tag already names the package, and a module path would go stale as
        soon as the class moved, which a fingerprint deliberately survives.
        """
        _check_nameable(type(self), self.tag)
        return {
            TAG_FIELD: self.tag,
            _VERSION_KEY: self.__version__,
            _PARAMS_KEY: encode(self._params(), mode=DOCUMENT),
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> Task:
        """
        Return the task a document describes.

        The tag is resolved through the model registry alone. A tag nothing
        registers raises rather than being imported: reading a document is
        not a reason to import whatever it names.
        """
        tag = document.get(TAG_FIELD)
        task_class = resolve_tagged_model(tag)
        if not issubclass(task_class, Task):
            msg = f"The tag {tag!r} names {task_class.__name__}, which is not a Task."
            raise ParameterError(msg)
        _check_version(task_class, document.get(_VERSION_KEY))
        params = {
            key: decode(value) for key, value in document.get(_PARAMS_KEY, {}).items()
        }
        return task_class(**params)

    def save(self, path: str | Path) -> Path:
        """
        Write this task to a file, and return where it was written.

        The suffix picks the format; see
        [`write_workflow`](`dascore.workflow.serialize.write_workflow`).
        """
        return write_workflow(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Task:
        """
        Return the task a file holds; see
        [`save`](`dascore.workflow.task.Task.save`).

        The task is named by its registered tag, so reading a file never
        imports whatever it happens to name -- which also means the class
        has to be defined at module level and already imported.
        """
        return cls.from_dict(read_workflow(Path(path)))

    def _params(self) -> dict[str, Any]:
        """Return this task's parameters, as the objects they are."""
        return model_values(self)

    def __eq__(self, other) -> bool:
        """Two tasks are equal if they are the same task, given the same."""
        # Not the base model's field comparison, which does not look at the
        # class: Hilbert(dim="time") and Envelope(dim="time") hold the same
        # fields and are not the same operation.
        if not isinstance(other, Task):
            return NotImplemented
        return type(self) is type(other) and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        """Hash a task the way it compares."""
        return hash(self.fingerprint)

    def __reduce__(self):
        """
        Pickle a task by its tag, and its parameters as themselves.

        A task class synthesized from a function is not reachable at any
        attribute path, so the default pickling cannot find it again in
        another process; its tag can. The parameters are left to pickle,
        which carries values a document cannot -- a function, a frame -- and
        carries an array as bytes rather than as a list of numbers.
        """
        _check_nameable(type(self), self.tag)
        return (_rebuild, (self.tag, self._params()))


def _rebuild(tag: str, params: dict[str, Any]) -> Task:
    """Return the task a tag and its parameters name; see `Task.__reduce__`."""
    task_class = resolve_tagged_model(tag)
    return task_class(**params)


def _check_nameable(task_class: type[Task], tag: str | None) -> None:
    """
    Refuse to write down a task whose class nothing could look up.

    A class defined inside a function is not registered, and one whose tag
    two classes claim is struck from the registry. Either way the failure
    belongs where the task is written, not in whoever reads it later.
    """
    if resolve_model_tag(tag or "") is not task_class:
        msg = (
            f"{task_class.__qualname__} is not registered under the tag "
            f"{tag!r}, so nothing could read it back. A task class has to "
            "be defined at module level, under a name no other class in "
            "its package claims."
        )
        raise ParameterError(msg)


def _check_version(task_class: type[Task], version: object) -> None:
    """Say so when a document was written by another version of a task."""
    # Not an error: an old document should still load and run. The version
    # has already done its work by changing the fingerprint, and refusing
    # the document would only hide what it was.
    if version is not None and version != task_class.__version__:
        msg = (
            f"The document holds {task_class.__name__} version {version!r}, "
            f"and this one is {task_class.__version__!r}. It will not "
            "fingerprint the way it did when it was written."
        )
        warnings.warn(msg, DASCoreWarning, stacklevel=3)


# The canonical instance of each task, held only while something else holds
# it: interning is for sharing one object, not for keeping it alive. A task's
# parameters can be as big as an array, so a cache which retained them would
# hold that memory long after the caller dropped it.
_INTERNED: weakref.WeakValueDictionary[str, Task] = weakref.WeakValueDictionary()


def intern(task: Task) -> Task:
    """
    Return one shared instance for every task which equals the given one.

    A patch function called in a loop builds an equal task every call.
    Interning them means the provenance of a thousand patches holds one task
    object rather than a thousand copies of it.

    Examples
    --------
    >>> from dascore.workflow import Task, intern
    >>> class ScaleExample(Task):
    ...     factor: float = 1.0
    >>> assert intern(ScaleExample()) is intern(ScaleExample())
    """
    existing = _INTERNED.get(task.fingerprint)
    # Two classes can only share a fingerprint if neither could be named in
    # a document, in which case they are left as the separate objects they
    # are rather than one standing in for the other.
    if existing is not None and type(existing) is type(task):
        return existing
    _INTERNED[task.fingerprint] = task
    return task


def own_arrays(values: Mapping) -> dict:
    """
    Return parameters with every array in them made read-only.

    A task is immutable and its fingerprint is cached on first use, so an
    array a caller kept writing to would leave the task reporting a
    fingerprint of values it no longer holds. Marking it read-only is what
    dascore does with a patch's data, and costs no copy -- which means the
    array marked is the caller's own, not a copy of it.
    """
    return {key: _own(value) for key, value in values.items()}


def _own(value: Any) -> Any:
    """Return one parameter value with the arrays inside it made read-only."""
    # Only array-likes: `array` would turn anything else, a string or a
    # tuple of bounds, into an array of its own.
    if is_array_like(value):
        return array(value)
    # Arrays reach a task inside a sequence as well -- a pair of masks, a
    # mapping of them -- and a container is walked rather than rebuilt when
    # it holds none.
    if isinstance(value, list | tuple):
        owned = [_own(x) for x in value]
        return type(value)(owned) if owned != list(value) else value
    if isinstance(value, Mapping):
        owned = {key: _own(x) for key, x in value.items()}
        return owned if owned != dict(value) else value
    return value


def _qualified_tag(cls: type) -> str:
    """Return the tag naming a class, always with its namespace."""
    # DASCore's own models register bare, so that a hand written file says
    # `object_type: Cable`. A fingerprint is not read by hand and is
    # compared against other packages' tasks, so it always names the
    # package the class came from.
    namespace = cls.__module__.split(".", 1)[0]
    return f"{namespace}{NAMESPACE_SEP}{cls.__name__}"


class FunctionTask(Task):
    """
    The behaviour a task synthesized from a function has.

    Kept apart from `make_function_task_class` so that the methods are
    ordinary, testable functions rather than closures built per class, and
    so that a synthesized class can be recognized by what it derives from.
    """

    # How the wrapped function is called: which fields go positionally --
    # with a flag for the one standing for a *args group, which is splatted
    # -- and which go by name. Worked out once, when the class is made.
    _original_function: ClassVar[Callable]
    _signature: ClassVar[inspect.Signature]
    _positional_names: ClassVar[tuple[tuple[str, bool], ...]]
    _keyword_names: ClassVar[tuple[str, ...]]

    def run(self, *args) -> Any:
        """Call the wrapped function with any inputs and this task's fields."""
        return self._original_function(*args, *self._call_args(), **self._call_kwargs())

    @classmethod
    def _from_call(cls, args: tuple, kwargs: dict) -> Self:
        """
        Return the task standing for one call of the wrapped function.

        The arguments are the ones the function was given, minus any input
        it is handed at run time, exactly as `_call_args` gives them back.

        Binding rather than validating: this runs on every call, and the
        function itself is what checks its own arguments.
        """
        bound = cls._signature.bind_partial(*args, **kwargs)
        # Not apply_defaults: model_construct fills a missing field from the
        # class default, which pydantic copies. Binding the function's own
        # default object instead would share a mutable one between tasks.
        values = dict(bound.arguments)
        # A **kwargs group arrives as one mapping and is spread back out,
        # which is how validation would have attached those names.
        for name, parameter in cls._signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                values |= values.pop(name, {})
        # model_construct skips validation, and with it the validator which
        # takes ownership of an array, so it is done here instead.
        return cls.model_construct(**own_arrays(values))

    def _call_args(self) -> tuple:
        """Return the arguments this task passes positionally."""
        out = []
        for name, packed in self._positional_names:
            value = getattr(self, name)
            if packed:
                out.extend(value)
            else:
                out.append(value)
        return tuple(out)

    def _call_kwargs(self) -> dict:
        """Return the arguments this task passes by name."""
        out = {name: getattr(self, name) for name in self._keyword_names}
        out.update(self.__pydantic_extra__ or {})
        return out


def task(func: Callable | None = None, *, version: str = "1.0", inputs: int = 1) -> Any:
    """
    Turn a function into a `Task` subclass.

    The function's parameters become the task's fields, except the first
    `inputs` of them, which are what the task is given when it runs.

    Parameters
    ----------
    func
        The function to convert.
    version
        The version of the new task class.
    inputs
        How many of the function's leading positional parameters are run
        time inputs rather than parameters. One by default, which is what a
        task handed a patch takes; zero makes a task which is fed nothing
        and is a source of its own.

    Examples
    --------
    >>> from dascore.workflow import task
    >>>
    >>> @task
    ... def scale_number_example(number, factor=1):
    ...     '''Scale a number.'''
    ...     return number * factor
    >>>
    >>> assert scale_number_example(factor=2).run(3) == 6
    >>>
    >>> @task(inputs=0)
    ... def source_number_example(value=1):
    ...     '''Make a number out of nothing.'''
    ...     return value
    >>>
    >>> assert source_number_example(value=2).run() == 2
    """

    def decorator(function: Callable) -> type[Task]:
        return make_function_task_class(function, version=version, inputs=inputs)

    return decorator if func is None else decorator(func)


def make_function_task_class(
    func: Callable,
    base: type[Task] = Task,
    version: str = "1.0",
    inputs: int = 0,
) -> type[Task]:
    """
    Return a `Task` subclass whose fields are a function's parameters.

    Parameters
    ----------
    func
        The function the class stands for.
    base
        The class to derive from; a patch function derives from
        `PatchProcessor` rather than from `Task` itself.
    version
        The version of the new class.
    inputs
        How many leading parameters are not made fields. They are what the
        task is given when it runs, such as the patch a patch function
        operates on. None by default here, unlike the `task` decorator,
        which takes one.
    """
    signature = inspect.signature(func)
    # A callable which is not a function -- a class, or an object with a
    # __call__ -- is named by its own class instead.
    qualname = getattr(func, "__qualname__", type(func).__qualname__)
    declared = list(signature.parameters.values())
    _check_inputs(func, declared, inputs)
    parameters = _remaining_parameters(declared, inputs)
    namespace, annotations = _build_fields(parameters)
    positional, keyword = _call_plan(parameters)
    # Declared ClassVar so that pydantic leaves them as class attributes
    # rather than making private, per-instance ones of them.
    annotations |= {
        "_original_function": ClassVar[Callable],
        "_signature": ClassVar[inspect.Signature],
        "_positional_names": ClassVar[tuple[tuple[str, bool], ...]],
        "_keyword_names": ClassVar[tuple[str, ...]],
    }
    namespace |= {
        "__annotations__": annotations,
        "__doc__": func.__doc__,
        "__module__": func.__module__,
        # Set here rather than after the class is built: the registry reads
        # both while `__init_subclass__` runs.
        "__qualname__": f"{qualname}.processor",
        "__version__": version,
        # A bare function in a class body is a method, and would be handed
        # the task as its first argument.
        "_original_function": staticmethod(func),
        # The signature the task's own parameters make, which is the one an
        # incoming call is bound against; the input a patch function is
        # given is not one of them.
        "_signature": signature.replace(parameters=parameters),
        "_positional_names": positional,
        "_keyword_names": keyword,
        "model_config": _build_config(base, parameters),
    }
    name = _camel_case(getattr(func, "__name__", type(func).__name__))
    # A field named for a BaseModel attribute -- `copy`, `json`, `schema` --
    # shadows it. That is what the caller asked for: the field is named by
    # the function, and the attribute it hides is not part of a task's API.
    with suppress_warnings(UserWarning, message="Field name .* shadows an attribute"):
        return type(name, (FunctionTask, base), namespace)


def _check_inputs(
    func: Callable, parameters: list[inspect.Parameter], inputs: int
) -> None:
    """
    Refuse a count of run time inputs the function could not be given.

    An input is passed positionally, so only a positional parameter can be
    one; a keyword only parameter, or a ``**kwargs`` group, stays with the
    task. A ``*args`` group takes as many as it is given, so a function
    which has one can be asked for any number. Refusing here puts the
    failure where the count is written rather than in the call which finds
    itself missing an argument.
    """
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    takeable = [x for x in parameters if x.kind in positional]
    packs_args = any(x.kind == inspect.Parameter.VAR_POSITIONAL for x in parameters)
    if inputs < 0 or (inputs > len(takeable) and not packs_args):
        name = getattr(func, "__name__", type(func).__name__)
        msg = (
            f"{name} takes {len(takeable)} arguments which could be an "
            f"input, and was asked for {inputs}. Spell the count it takes, "
            "such as inputs=0 for a task which is handed nothing."
        )
        raise ParameterError(msg)


def _remaining_parameters(
    declared: list[inspect.Parameter], inputs: int
) -> list[inspect.Parameter]:
    """
    Return the parameters left once the run time inputs are taken out.

    Taken off the front one at a time rather than sliced by count: a
    ``*args`` group absorbs every input which is left, and whatever is
    declared after it -- a keyword only parameter, a ``**kwargs`` group --
    is still the task's, which slicing by count would drop.
    """
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    out: list[inspect.Parameter] = []
    taken = 0
    for parameter in declared:
        if taken < inputs and parameter.kind in positional:
            taken += 1
        elif taken < inputs and parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            taken = inputs
        else:
            out.append(parameter)
    return out


def _call_plan(
    parameters: list[inspect.Parameter],
) -> tuple[tuple[tuple[str, bool], ...], tuple[str, ...]]:
    """
    Return which fields are passed positionally and which by name.

    Everything which can go by name does, which keeps a call readable and
    lets a function reorder its keyword parameters without changing what a
    stored task means.
    """
    kinds = {x.kind for x in parameters}
    # Nothing can be passed by name before a *args group without landing in
    # the group instead.
    packs_args = inspect.Parameter.VAR_POSITIONAL in kinds
    positional, keyword = [], []
    for parameter in parameters:
        kind = parameter.kind
        if kind == inspect.Parameter.VAR_KEYWORD:
            # Its values are extras rather than a field, and are spread out
            # by `_call_kwargs`.
            continue
        if kind == inspect.Parameter.VAR_POSITIONAL:
            positional.append((parameter.name, True))
        elif kind == inspect.Parameter.POSITIONAL_ONLY or (
            kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and packs_args
        ):
            positional.append((parameter.name, False))
        else:
            keyword.append(parameter.name)
    return tuple(positional), tuple(keyword)


def _build_fields(
    parameters: list[inspect.Parameter],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the namespace and annotations a function's parameters make."""
    namespace: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # The extras a **kwargs group collects are not fields; the class
            # is configured to keep them instead.
            continue
        # Every field is typed Any: the function validates its own
        # arguments, and a patch function's annotations name forward
        # references (PatchType) which pydantic cannot resolve here.
        annotations[parameter.name] = Any
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            namespace[parameter.name] = ()
        elif parameter.default is not inspect.Parameter.empty:
            namespace[parameter.name] = _resolve_default(parameter.default)
    return namespace, annotations


def _resolve_default(default: Any) -> Any:
    """Return the value a parameter defaults to."""
    # A function which spells its default `x=Field(default=3)` means 3; the
    # FieldInfo itself would otherwise become the default value.
    if isinstance(default, FieldInfo):
        return default.get_default(call_default_factory=True)
    return default


def _build_config(base: type[Task], parameters: list[inspect.Parameter]) -> ConfigDict:
    """Return the model config a function's parameters call for."""
    kinds = {x.kind for x in parameters}
    packs_kwargs = inspect.Parameter.VAR_KEYWORD in kinds
    # A function taking **kwargs is called with names its signature does not
    # list -- `patch.pad(time=(2, 3))` -- so its task has to keep them.
    return ConfigDict(
        **(base.model_config | {"extra": "allow" if packs_kwargs else "forbid"})
    )


def _camel_case(name: str) -> str:
    """Return the class name a function name makes."""
    return "".join(part.title() for part in name.split("_"))
