"""Run exactly one anchor job in this process, then exit.

Process-per-job isolation. Surge segfaults randomly under concurrency, and
inside a ProcessPoolExecutor a dead child breaks the whole pool — every
in-flight job dies with it. Measured live: crashes every ~3.5 min against
~7 min jobs, so long jobs were killed and restarted forever (throughput
collapsed from 4.16 to 0.29 shards/min).

Here a crash is just a non-zero exit code that affects nothing else. The
cost is rebuilding the Surge host and its parameter map per job (~19 s,
about 4% of a job) — cheap insurance against a livelock.

Protocol: job JSON on stdin, result JSON on stdout prefixed by RESULT_TAG.
"""

from __future__ import annotations

import json
import sys

RESULT_TAG = "__TGLAB_RESULT__"


def main() -> int:
    job = json.loads(sys.stdin.read())
    from timbre_graph_lab import gen
    from timbre_graph_lab.config import LabConfig

    cfg = LabConfig()
    gen._init_worker(cfg)  # noqa: SLF001 — this module IS the worker
    result = gen.process_anchor(job)
    sys.stdout.write(RESULT_TAG + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
