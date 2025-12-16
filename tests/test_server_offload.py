import concurrent.futures
from pathlib import Path

import sigil_mcp.server as server


def test_on_file_change_uses_executor(monkeypatch, tmp_path):
    repo_name = "r1"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    server.REPOS[repo_name] = repo_root

    called = []

    # Replace process_pool with a ThreadPoolExecutor for test determinism
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(server, "process_pool", ex)
    monkeypatch.setattr(server, "USE_PROCESS_POOL", True)

    # Replace the worker task with a simple function that records invocation
    def fake_task(cfg_json, rn, rr, fp):
        called.append((rn, rr, fp))
        return True

    monkeypatch.setattr(server, "index_file_task", fake_task)

    # Minimal stub index to satisfy embed init checks
    class StubIndex:
        embed_fn = None

    monkeypatch.setattr(server, "_get_index", lambda: StubIndex())

    file_path = repo_root / "a.py"
    file_path.write_text("print(1)\n")

    server._on_file_change(repo_name, file_path, "modified")

    # Allow background worker to run
    ex.shutdown(wait=True)

    assert called, "Worker task was not invoked"
    rn, rr, fp = called[0]
    assert rn == repo_name
    assert Path(rr) == repo_root
    assert Path(fp) == file_path
