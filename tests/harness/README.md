Run the offload harness

This harness exercises the `sigil_mcp.server` offload path for manual integration
testing. It replaces the heavy `index_file_task` with a lightweight stub and uses a
thread-backed executor for deterministic behavior.

Usage:

```bash
python tests/harness/run_offload_harness.py
```

The script will print a success/failure message and exit with code 0 on success.

Note: This is a convenience tool for developers and CI; it does not replace the
full test suite. It intentionally avoids constructing a full `SigilIndex` in the
worker process to keep runs fast and deterministic.
