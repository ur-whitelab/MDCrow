import json
import os

import pytest
from langchain_openai import ChatOpenAI

from mdcrow.agent.agent import MDCrow
from mdcrow.agent.memory import MemoryManager


@pytest.fixture
def memory_manager(get_registry):
    llm = ChatOpenAI()
    return MemoryManager(get_registry("raw", False), llm)


def test_mdcrow_memory():
    mdcrow_memory = MDCrow(use_memory=True)
    mdcrow_no_memory = MDCrow(use_memory=False)
    assert mdcrow_memory.use_memory is True
    assert mdcrow_no_memory.use_memory is False

    mdcrow_memory = MDCrow(use_memory=True, run_id="TESTRUNN")
    assert mdcrow_memory.run_id == "TESTRUNN"

    mdcrow_memory = MDCrow(use_memory=True, run_id="")
    assert mdcrow_memory.run_id


def test_memory_init(memory_manager, get_registry):
    llm = ChatOpenAI()

    assert memory_manager is not None
    assert memory_manager.run_id is not None
    assert len(memory_manager.run_id) == 8

    mm_path_id = MemoryManager(get_registry("raw", False), llm, run_id="TESTRUNN")
    assert mm_path_id.run_id == "TESTRUNN"


def test_force_clear_mem(monkeypatch):
    dummy_test_dir = "ckpt_test"

    mdcrow = MDCrow(ckpt_dir=dummy_test_dir)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    print(mdcrow.path_registry.ckpt_dir)
    print(mdcrow.path_registry.json_file_path)
    print(os.path.dirname(mdcrow.path_registry.ckpt_dir))
    mdcrow.force_clear_mem(all=False)
    assert not os.path.exists(mdcrow.path_registry.ckpt_dir)
    assert not os.path.exists(mdcrow.path_registry.json_file_path)
    assert os.path.exists(
        os.path.basename(os.path.dirname(mdcrow.path_registry.ckpt_dir))
    )

    mdcrow = MDCrow(ckpt_dir=dummy_test_dir)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    mdcrow.force_clear_mem(all=True)
    print(mdcrow.path_registry.ckpt_dir)
    print(mdcrow.path_registry.json_file_path)
    print(os.path.dirname(mdcrow.path_registry.ckpt_dir))
    assert not os.path.exists(mdcrow.path_registry.ckpt_dir)
    assert not os.path.exists(mdcrow.path_registry.json_file_path)
    assert not os.path.exists(os.path.dirname(mdcrow.path_registry.ckpt_dir))


def test_write_to_json_new_file(tmp_path, memory_manager):
    file_path = tmp_path / "test.json"
    test_data = {"key": "value"}
    memory_manager._write_to_json(test_data, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == test_data


def test_write_to_json_existing_file(tmp_path, memory_manager):
    file_path = tmp_path / "test.json"
    initial_data = {"initial_key": "initial_value"}
    update_data = {"updated_key": "updated_value"}
    with open(file_path, "w") as f:
        json.dump(initial_data, f)

    memory_manager._write_to_json(update_data, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == {**initial_data, **update_data}
