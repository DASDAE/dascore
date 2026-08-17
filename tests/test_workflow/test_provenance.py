"""Tests for the two records of where data came from."""

from __future__ import annotations

import pickle

import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.workflow import Provenance, ProvenanceNode, SourceInfo, Task


class StepTask(Task):
    """A step which changes a number."""

    value: int = 1

    def run(self, number):
        """Add to a number."""
        return number + self.value


class MergeTask(Task):
    """A step which takes several numbers."""

    def run(self, *numbers):
        """Add up numbers."""
        return sum(numbers)


class LaterTask(Task):
    """A step which runs after another, and is named differently."""

    def run(self, number):
        """Double a number."""
        return number * 2


@pytest.fixture
def source_node():
    """A node standing for a patch read from a file."""
    return ProvenanceNode(
        source=SourceInfo(format="DASDAE", path="/data/first.h5", key="patch_2"),
        patch_id="first",
    )


@pytest.fixture
def chain(source_node):
    """Two steps run one after the other on one source."""
    first = ProvenanceNode(
        task=StepTask(value=1),
        parents=(source_node,),
        input_pairs=(("first", ""),),
        patch_id="first",
        processing_id="one",
    )
    return ProvenanceNode(
        task=LaterTask(),
        parents=(first,),
        input_pairs=(("first", "one"),),
        patch_id="first",
        processing_id="two",
        backend="numpy",
    )


@pytest.fixture
def merged(source_node):
    """One step over patches read from three files."""
    others = [
        ProvenanceNode(source=SourceInfo(path=f"/data/{x}.h5"), patch_id=x)
        for x in ("second", "third")
    ]
    return ProvenanceNode(
        task=MergeTask(),
        parents=(source_node, *others),
        patch_id="merged",
        processing_id="three",
    )


class TestWalk:
    """Tests for reading a graph."""

    def test_walk_reaches_everything(self, chain):
        """Every node behind one is reachable from it."""
        assert len(list(chain.walk())) == 3

    def test_walk_is_oldest_first(self, chain):
        """A node is walked after the nodes which fed it."""
        walked = list(chain.walk())
        assert walked[0].source is not None
        assert walked[-1] is chain

    def test_walk_yields_each_node_once(self, source_node):
        """A node two branches share is walked once."""
        left = ProvenanceNode(task=StepTask(value=1), parents=(source_node,))
        right = ProvenanceNode(task=StepTask(value=2), parents=(source_node,))
        merged = ProvenanceNode(task=MergeTask(), parents=(left, right))
        assert len(list(merged.walk())) == 4

    def test_steps(self, chain):
        """The steps are the nodes which did something, oldest first."""
        assert [type(x.task).__name__ for x in chain.steps()] == [
            "StepTask",
            "LaterTask",
        ]

    def test_sources(self, merged):
        """The sources are where the data was read from."""
        assert [x.path for x in merged.sources()] == [
            "/data/first.h5",
            "/data/second.h5",
            "/data/third.h5",
        ]

    def test_a_source_has_no_steps(self, source_node):
        """Nothing was done to a patch which was only read."""
        assert source_node.steps() == ()


class TestDescribe:
    """Tests for the readable listing which replaces a history string."""

    def test_lists_the_steps(self, chain):
        """Every step is named, oldest first."""
        assert chain.describe().splitlines()[1:] == ["StepTask", "LaterTask"]

    def test_names_the_source(self, chain):
        """The file the data came from is the first line."""
        assert chain.describe().splitlines()[0] == "read DASDAE /data/first.h5"

    def test_collapses_many_sources(self, merged):
        """A patch built from many files does not list them all."""
        assert merged.describe().splitlines()[0] == "read 3 sources"

    def test_says_how_wide_a_step_was(self, merged):
        """A step over several inputs says how many."""
        assert "MergeTask over 3 inputs" in merged.describe()


class TestToPipe:
    """Tests for turning a graph back into something runnable."""

    def test_chain(self, chain):
        """A chain of steps becomes a pipe of the same tasks."""
        pipe = chain.to_pipe()
        assert len(pipe) == 2
        assert pipe.run(0) == 2

    def test_sources_become_the_pipes_input(self, merged):
        """A step fed only by files takes its inputs from the caller."""
        pipe = merged.to_pipe()
        assert len(pipe) == 1
        assert pipe.run(1, 2, 3) == 6

    def test_repeated_task(self, source_node):
        """The same task twice stays two steps."""
        first = ProvenanceNode(task=StepTask(value=1), parents=(source_node,))
        second = ProvenanceNode(task=StepTask(value=1), parents=(first,))
        assert len(second.to_pipe()) == 2

    def test_nothing_was_done(self, source_node):
        """A patch which was only read has no pipe."""
        with pytest.raises(ParameterError, match="no pipe"):
            source_node.to_pipe()

    def test_two_wide_sources_refused(self, source_node):
        """Two steps reading straight from files cannot both be fed."""
        others = [ProvenanceNode(source=SourceInfo(path=f"/{x}.h5")) for x in "abc"]
        first = ProvenanceNode(task=MergeTask(), parents=(source_node, others[0]))
        second = ProvenanceNode(task=MergeTask(), parents=tuple(others[1:]))
        both = ProvenanceNode(task=MergeTask(), parents=(first, second))
        with pytest.raises(ParameterError, match="no way to describe"):
            both.to_pipe()

    def test_mixed_inputs_refused(self, source_node):
        """A step fed by a step and a file at once cannot be described."""
        step = ProvenanceNode(task=StepTask(value=1), parents=(source_node,))
        other = ProvenanceNode(source=SourceInfo(path="/data/other.h5"))
        mixed = ProvenanceNode(task=MergeTask(), parents=(step, other))
        with pytest.raises(ParameterError, match="no way to describe"):
            mixed.to_pipe()


class TestNodeDocuments:
    """Tests for writing a graph down and reading it back."""

    def test_round_trip(self, chain):
        """A graph read back holds the same steps."""
        rebuilt = ProvenanceNode.from_json(chain.to_json())
        assert rebuilt.to_pipe() == chain.to_pipe()
        assert rebuilt.processing_id == chain.processing_id

    def test_shape_kept(self, merged):
        """A graph read back has the shape it was written with."""
        rebuilt = ProvenanceNode.from_json(merged.to_json())
        assert len(rebuilt.parents) == 3
        assert [x.path for x in rebuilt.sources()] == [x.path for x in merged.sources()]

    def test_shared_node_stays_shared(self, source_node):
        """A node two branches share is written once and read back once."""
        left = ProvenanceNode(task=StepTask(value=1), parents=(source_node,))
        right = ProvenanceNode(task=StepTask(value=2), parents=(source_node,))
        merged = ProvenanceNode(task=MergeTask(), parents=(left, right))
        rebuilt = ProvenanceNode.from_json(merged.to_json())
        assert len(list(rebuilt.walk())) == 4

    def test_ids_kept(self, chain):
        """The ids a step produced survive the trip."""
        rebuilt = ProvenanceNode.from_json(chain.to_json())
        assert rebuilt.patch_id == chain.patch_id
        assert rebuilt.input_pairs == chain.input_pairs

    def test_source_node_round_trip(self, source_node):
        """A graph of one source, with no steps, still round trips."""
        rebuilt = ProvenanceNode.from_json(source_node.to_json())
        assert rebuilt.source == source_node.source
        assert rebuilt.source.key == "patch_2"

    def test_everything_a_node_holds_round_trips(self, chain):
        """Every field a node holds survives the trip."""
        rebuilt = ProvenanceNode.from_json(chain.to_json())
        for name in type(chain).model_fields:
            if name != "parents":
                assert getattr(rebuilt, name) == getattr(chain, name)

    def test_pickle(self, chain):
        """A graph survives a pickle."""
        rebuilt = pickle.loads(pickle.dumps(chain))
        assert rebuilt.to_pipe() == chain.to_pipe()

    def test_equal_after_a_round_trip(self, chain):
        """A node read back is the step it was, and hashes as one."""
        rebuilt = ProvenanceNode.from_json(chain.to_json())
        assert rebuilt == chain
        assert len({rebuilt, chain}) == 1

    def test_different_steps_differ(self, chain, source_node):
        """Two nodes standing for different steps are not each other."""
        other = ProvenanceNode(
            task=StepTask(value=99),
            parents=(source_node,),
            patch_id=chain.patch_id,
            processing_id=chain.processing_id,
        )
        assert other != chain
        assert len({other, chain}) == 2

    def test_not_a_node(self, chain):
        """Comparison with something else is left to the something else."""
        assert chain.__eq__("a node") is NotImplemented


class TestProvenance:
    """Tests for the durable record of a run."""

    @pytest.fixture
    def provenance(self, chain):
        """A record of the pipe a graph describes."""
        return chain.to_pipe().get_provenance(operator="a test")

    def test_records_the_pipe(self, provenance, chain):
        """The record holds the pipe it was made from."""
        assert provenance.pipe == chain.to_pipe()

    def test_fingerprint_is_the_pipes(self, provenance):
        """A record is identified by the pipe it holds."""
        assert provenance.fingerprint == provenance.pipe.fingerprint

    def test_records_the_run(self, provenance):
        """The record says which DASCore and which machine ran it."""
        assert provenance.dascore_version == dc.__version__
        assert provenance.python_version
        assert "platform" in provenance.system_info

    def test_metadata(self, provenance):
        """Anything else worth recording is kept."""
        assert provenance.metadata["operator"] == "a test"

    @pytest.mark.parametrize("name", ["run.json", "run.yaml"])
    def test_save_and_load(self, provenance, tmp_path, name):
        """A record read back from a file describes the same run."""
        rebuilt = Provenance.load(provenance.save(tmp_path / name))
        assert rebuilt.pipe == provenance.pipe
        assert rebuilt.created_at == provenance.created_at
        assert rebuilt.metadata == provenance.metadata

    def test_source_provenance(self, provenance):
        """A record of a run over another run's results holds both."""
        document = provenance.to_dict() | {"source_provenance": [provenance.to_dict()]}
        rebuilt = Provenance.from_dict(document)
        assert rebuilt.source_provenance[0].fingerprint == provenance.fingerprint

    def test_hashed_by_its_pipe(self, provenance, chain):
        """Two records of one pipe land in the same place in a dict."""
        again = Provenance.from_pipe(chain.to_pipe(), operator="someone else")
        assert hash(again) == hash(provenance)

    def test_metadata_cannot_be_edited(self, provenance):
        """A durable record is durable: nothing about it can be changed."""
        with pytest.raises(TypeError):
            provenance.metadata["sneaked"] = 1

    def test_new_fields_are_written(self, provenance):
        """Every field the record holds reaches its document."""
        written = set(provenance.to_dict())
        assert set(type(provenance).model_fields) <= written
