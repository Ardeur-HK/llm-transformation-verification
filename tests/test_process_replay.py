"""Exercise replay with generated local evidence and isolated manipulations."""

from __future__ import annotations

import shutil

import pytest
import safetensors.torch
import torch

from evidence_audit.checkers import process_replay
from evidence_audit.evidence import recorder
from examples.replay_demo import BATCH_SIZE, END_STEP, START_STEP, record_fixture


@pytest.fixture(scope="module")
def honest_run(tmp_path_factory):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_dir = tmp_path_factory.mktemp("replay") / "honest"
        record_fixture(run_dir)
        yield run_dir
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def staged_run(honest_run, tmp_path):
    return shutil.copytree(honest_run, tmp_path / "staged")


def replay(run_dir, **kwargs):
    return process_replay.replay_interval(run_dir, START_STEP, END_STEP, **kwargs)


def boundary(run_dir, step):
    return run_dir / "boundaries" / f"step{step:04d}"


def perturb_tensor(path, key=None):
    tensors = safetensors.torch.load_file(str(path))
    key = min(tensors) if key is None else key
    tensors[key][0, 0] += 0.01
    safetensors.torch.save_file(tensors, str(path))
    return key


def assert_tensor_rejection(result):
    assert result["status"] == "rejected", result
    assert result["execution_status"] == "ok"
    return result["comparison"]


def test_honest_replay(honest_run):
    result = replay(honest_run)
    assert result["status"] == "supported", result
    assert result["execution_status"] == "ok"
    assert result["tolerance_policy"] == "bitwise_fp32_v0"
    assert result["interval"]["updates"] == [1, 2, 3]
    assert result["interval"]["batch_rows"] == [2, 8]
    assert result["coverage"]["unverified_intervals"] == ["[0,1)", "[4,5)"]
    comparison = result["comparison"]
    assert comparison["adapter_tensors"] == 4
    assert comparison["optimizer_slot_tensors"] == 12
    assert comparison["adapter_equal"]
    assert comparison["optimizer_slots_equal"]
    assert comparison["end_loss_equal"]
    assert comparison["replay_losses_match_train_log"]
    assert result["restore"]["rng_restored"]
    assert result["restore"]["optimizer_slots_match_start"]
    assert result["restore"]["buffers_match_step0"]


def test_evidence_manifest_covers_generated_files(honest_run):
    manifest = recorder.load_json(honest_run / recorder.MANIFEST_NAME)
    files = {
        path.relative_to(honest_run).as_posix(): path
        for path in honest_run.rglob("*")
        if path.is_file() and path.name != recorder.MANIFEST_NAME
    }
    assert set(manifest["files"]) == set(files)
    for name, path in files.items():
        assert manifest["files"][name] == {
            "sha256": recorder.sha256_file(path),
            "bytes": path.stat().st_size,
        }


def test_swapped_batches_reject(staged_run):
    path = staged_run / "batches.safetensors"
    tensors = safetensors.torch.load_file(str(path))
    start = START_STEP * BATCH_SIZE
    second_batch = start + BATCH_SIZE
    stop = second_batch + BATCH_SIZE
    for tensor in tensors.values():
        tensor[start:stop] = torch.cat(
            (tensor[second_batch:stop], tensor[start:second_batch])
        )
    safetensors.torch.save_file(tensors, str(path))

    comparison = assert_tensor_rejection(replay(staged_run))
    assert comparison["adapter_mismatched"]
    assert comparison["optimizer_slot_mismatched"]
    assert not comparison["replay_losses_match_train_log"]


def test_conflicting_learning_rate_reports_specific_error(staged_run):
    path = boundary(staged_run, START_STEP) / "optimizer.json"
    meta = recorder.load_json(path)
    meta["param_groups"][0]["lr"] *= 2
    recorder.save_json(path, meta)

    result = replay(staged_run)
    assert result["status"] == "error", result
    assert result["comparison"] is None
    assert "optimizer meta group 0 lr=" in result["error"]
    assert "contradicts recorded hyperparameters" in result["error"]


def test_altered_start_optimizer_slot_rejects(staged_run):
    path = boundary(staged_run, START_STEP) / "optimizer.safetensors"
    slot = perturb_tensor(path, "p0.exp_avg_sq")

    comparison = assert_tensor_rejection(replay(staged_run))
    assert slot in comparison["optimizer_slot_mismatched"]
    assert comparison["adapter_mismatched"]


def test_altered_start_adapter_rejects(staged_run):
    path = boundary(staged_run, START_STEP) / "adapter.safetensors"
    key = perturb_tensor(path)

    comparison = assert_tensor_rejection(replay(staged_run))
    assert key in comparison["adapter_mismatched"]
    assert comparison["optimizer_slot_mismatched"]


def test_altered_end_adapter_rejects_only_changed_tensor(staged_run):
    path = boundary(staged_run, END_STEP) / "adapter.safetensors"
    key = perturb_tensor(path)

    comparison = assert_tensor_rejection(replay(staged_run))
    assert comparison["adapter_mismatched"] == [key]
    assert comparison["optimizer_slots_equal"]
    assert comparison["replay_losses_match_train_log"]


def test_changed_loss_and_buffer_records_are_diagnostics(staged_run):
    path = boundary(staged_run, END_STEP) / "state.json"
    state = recorder.load_json(path)
    state["loss_hex"] = float(0).hex()
    recorder.save_json(path, state)
    buffers = boundary(staged_run, 0) / "buffers.safetensors"
    tensors = safetensors.torch.load_file(str(buffers))
    tensors["unexpected_buffer"] = torch.zeros(1)
    safetensors.torch.save_file(tensors, str(buffers))

    result = replay(staged_run)
    assert result["status"] == "supported", result
    assert not result["comparison"]["end_loss_equal"]
    assert not result["restore"]["buffers_match_step0"]


def test_invalid_optimizer_slot_key_reports_error(staged_run):
    path = boundary(staged_run, START_STEP) / "optimizer.safetensors"
    tensors = safetensors.torch.load_file(str(path))
    tensors["p-1.exp_avg_sq"] = tensors.pop("p0.exp_avg_sq")
    safetensors.torch.save_file(tensors, str(path))

    result = replay(staged_run)
    assert result["status"] == "error"
    assert "invalid optimizer slot key 'p-1.exp_avg_sq'" in result["error"]


def test_malformed_json_reports_error(staged_run):
    (staged_run / "train_log.json").write_text("{", encoding="utf-8")
    result = replay(staged_run)
    assert result["status"] == "error"
    assert result["execution_status"] == "error"
    assert "JSONDecodeError" in result["error"]
    assert result["comparison"] is None


def test_local_snapshot_override(staged_run, tmp_path):
    path = staged_run / "run_metadata.json"
    metadata = recorder.load_json(path)
    metadata["base_model"]["snapshot_path"] = str(tmp_path / "missing-base")
    recorder.save_json(path, metadata)

    missing = replay(staged_run)
    assert missing["status"] == "error"
    assert "local base model snapshot is not a directory" in missing["error"]
    relocated = replay(staged_run, base_snapshot=staged_run / "base")
    assert relocated["status"] == "supported", relocated
