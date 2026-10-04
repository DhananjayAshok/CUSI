"""
Smoke test: AndroidWorld's run.py -> *_cusi agent -> CUSI vLLMModel -> HTTP -> agent.

Runs without an emulator or GPU. A local mock OpenAI-compatible server stands in for
vLLM, AndroidWorld's FakeAsyncEnv stands in for the emulator, and a stub adb satisfies
run.py's import-time adb lookup. Run from the CUSI root with CUSI on PYTHONPATH:

    source scripts/utils.sh && python tests/smoke_android_world_cusi.py
"""
import os
import sys
import tempfile
from unittest import mock
from tests.mock_vllm import start_mock_vllm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLY = "Reason: done.\nAction: {\"action_type\": \"status\", \"goal_status\": \"complete\"}"


def count_images(*, request):
    content = request["body"]["messages"][0]["content"]
    return sum(1 for part in content if part["type"] == "image_url")


def run_agent(*, run_module, agent_name, base_url, requests):
    from absl import flags
    from android_world.env import adb_utils
    from android_world.utils import test_utils

    flags.FLAGS.unparse_flags()
    flags.FLAGS([
        "run.py", f"--agent_name={agent_name}", "--model_backend=vllm",
        "--model_name=mock-model", f"--vllm_base_url={base_url}",
    ])
    with mock.patch.object(adb_utils, "get_orientation", return_value=0), \
         mock.patch.object(adb_utils, "get_physical_frame_boundary", return_value=[0, 0, 100, 100]):
        agent = run_module._get_agent(test_utils.FakeAsyncEnv())
        n_before = len(requests)
        result = agent.step("do something")
    raw = result.data["action_raw_response"]
    request = requests[n_before]
    assert result.done, f"{agent_name}: agent did not finish"
    assert request["path"] == "/v1/chat/completions", request["path"]
    assert request["body"]["model"] == "mock-model", request["body"]["model"]
    assert raw["output"] == REPLY, raw
    assert raw["meta"] == {"input_tokens": 11, "output_tokens": 7}, raw["meta"]
    print(f"{agent_name}: OK  done={result.done}  images_sent={count_images(request=request)}  "
          f"requests={len(requests) - n_before}  meta={raw['meta']}")
    return count_images(request=request)


def main():
    server, base_url, requests = start_mock_vllm(reply=REPLY)

    # run.py looks for adb at import time; a stub on PATH satisfies it.
    stub_dir = tempfile.mkdtemp()
    stub_adb = os.path.join(stub_dir, "adb")
    with open(stub_adb, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(stub_adb, 0o755)
    os.environ["PATH"] = stub_dir + os.pathsep + os.environ["PATH"]

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "android_world"))
    import run as run_module

    m3a_images = run_agent(run_module=run_module, agent_name="m3a_cusi", base_url=base_url, requests=requests)
    t3a_images = run_agent(run_module=run_module, agent_name="t3a_cusi", base_url=base_url, requests=requests)
    assert m3a_images == 2, f"M3A should send 2 screenshots, sent {m3a_images}"
    assert t3a_images == 0, f"T3A is text only, sent {t3a_images} images"
    server.shutdown()
    print("ALL OK")


if __name__ == "__main__":
    main()
