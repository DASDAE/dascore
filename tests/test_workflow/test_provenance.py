"""Tests for the two records of where data came from."""

from __future__ import annotations

import pickle

import numpy as np
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


class SourceTask(Task):
    """A step which made a number out of nothing."""

    value: int = 1

    def run(self):
        """Return the number."""
        return self.value


class ArrayTask(Task):
    """A step parametrized by an array, which a document has to carry."""

    weights: object = None

    def run(self, number):
        """Hand back what it was given."""
        return number


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


class TestNodeIdentity:
    """Tests for when two nodes stand for the same step."""

    def test_the_same_step_on_different_data(self):
        """Two nodes reading different files are two steps, not one."""
        first = ProvenanceNode(source=SourceInfo(path="/first.h5"))
        second = ProvenanceNode(source=SourceInfo(path="/second.h5"))
        left = ProvenanceNode(task=StepTask(value=1), parents=(first,))
        right = ProvenanceNode(task=StepTask(value=1), parents=(second,))
        assert left != right
        assert len({left, right}) == 2

    def test_the_same_step_on_the_same_data(self):
        """A graph rebuilt from the same lineage holds the same steps."""
        source = ProvenanceNode(source=SourceInfo(path="/first.h5"))
        left = ProvenanceNode(task=StepTask(value=1), parents=(source,))
        right = ProvenanceNode(task=StepTask(value=1), parents=(source,))
        assert left == right
        assert len({left, right}) == 1

    def test_a_step_further_back(self):
        """Two nodes differ when anything behind them does."""
        source = ProvenanceNode(source=SourceInfo(path="/first.h5"))
        left = ProvenanceNode(task=StepTask(value=1), parents=(source,))
        right = ProvenanceNode(task=StepTask(value=2), parents=(source,))
        after_left = ProvenanceNode(task=LaterTask(), parents=(left,))
        after_right = ProvenanceNode(task=LaterTask(), parents=(right,))
        assert after_left != after_right


class TestToPipeRefusals:
    """Tests for the graphs which have no pipe."""

    def test_steps_which_never_meet(self):
        """Two chains joined by a step which recorded no task have no pipe."""
        source = ProvenanceNode()
        left = ProvenanceNode(task=StepTask(value=2), parents=(source,))
        right = ProvenanceNode(task=StepTask(value=3), parents=(source,))
        joined = ProvenanceNode(parents=(left, right))
        with pytest.raises(ParameterError, match="could run"):
            joined.to_pipe()

    def test_sources_read_by_steps_of_different_widths(self):
        """A pipe hands each source one input, so it cannot say this."""
        made = ProvenanceNode(task=StepTask(value=9))
        chunked = ProvenanceNode(
            task=StepTask(value=2), parents=(ProvenanceNode(), ProvenanceNode())
        )
        merged = ProvenanceNode(task=StepTask(value=3), parents=(made, chunked))
        with pytest.raises(ParameterError, match="the same one input"):
            merged.to_pipe()

    def test_sources_which_make_their_own_values(self):
        """
        Steps which took nothing have a pipe, which is run with nothing.

        A pipe hands each of its sources the same thing, and nothing is as
        good as one input each -- which is the shape `Pipe` already runs.
        """
        first = ProvenanceNode(task=SourceTask(value=1))
        second = ProvenanceNode(task=SourceTask(value=2))
        merged = ProvenanceNode(task=MergeTask(), parents=(first, second))
        assert merged.to_pipe().run() == 3

    def test_an_input_whose_node_was_not_kept(self):
        """A step is fed from more places than the pipe can name."""
        source = ProvenanceNode()
        earlier = ProvenanceNode(task=StepTask(value=2), parents=(source,))
        node = ProvenanceNode(
            task=StepTask(value=3),
            parents=(earlier,),
            input_pairs=(("one", ""), ("two", "")),
        )
        with pytest.raises(ParameterError, match="straight from a source"):
            node.to_pipe()


class TestProvenanceDocuments:
    """Tests for what a record of a run can be written with."""

    def test_a_task_holding_an_array(self):
        """
        A record writes its pipe the way a pipe does.

        Dumped by pydantic instead it would raise on an array parameter --
        which the pipes most worth recording hold -- before reaching the
        line that replaces what it produced.
        """
        pipe = ArrayTask(weights=np.arange(3.0)) | ArrayTask(weights=np.ones(2))
        rebuilt = Provenance.from_dict(pipe.get_provenance().to_dict())
        assert rebuilt.pipe == pipe

    def test_metadata_a_document_has_no_shape_for(self):
        """Metadata is whatever was worth recording, and comes back as it."""
        when = np.datetime64("2020-01-01")
        record = Provenance.from_pipe(
            StepTask(value=1) | StepTask(value=2), when=when, run=3
        )
        rebuilt = Provenance.from_dict(record.to_dict())
        assert rebuilt.metadata["when"] == when
        assert rebuilt.metadata["run"] == 3

    def test_document_of_something_else(self):
        """A document which records no run says so."""
        with pytest.raises(ParameterError, match="states the pipe it ran"):
            Provenance.from_dict({"dascore_version": "1.0"})


class TestProvenanceNodeDocuments:
    """Tests for reading a graph back from text which may not hold one."""

    def test_text_which_does_not_parse(self):
        """Which says so as a workflow document does, naming what it read."""
        with pytest.raises(ParameterError, match="Could not parse JSON"):
            ProvenanceNode.from_json("{not json at all")

    def test_a_node_missing_a_field(self):
        """A document which parses but is not a graph says so."""
        with pytest.raises(ParameterError, match="no graph of what was done"):
            ProvenanceNode.from_json('{"nodes": [{}], "output": 0}')

    def test_an_output_which_is_not_there(self):
        """Nor is one whose result names a node it did not write."""
        with pytest.raises(ParameterError, match="no graph of what was done"):
            ProvenanceNode.from_json('{"nodes": [], "output": 3}')

    def test_text_holding_something_else(self):
        """A document which is not a graph is refused, not indexed into."""
        with pytest.raises(ParameterError, match="no record of what was done"):
            ProvenanceNode.from_json('{"tasks": {}}')


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
        if name.endswith(".yaml"):
            # pyyaml is an optional install.
            pytest.importorskip("yaml")
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
