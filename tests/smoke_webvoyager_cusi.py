"""
Smoke test of WebVoyager's run.py and auto_eval.py against a mock vLLM server, with no Chrome or GPU.
    python tests/smoke_webvoyager_cusi.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib
from PIL import Image
from tests.mock_vllm import start_mock_vllm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBVOYAGER = os.path.join(PROJECT_ROOT, "WebVoyager")


def write_pdf(*, path, text):
    """Write a one-page PDF containing text."""
    stream = zlib.compress(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)


def check_run_py(*, base_url, requests, work):
    sys.path.insert(0, WEBVOYAGER)
    import run as wv_run

    # An empty task file: main() parses every flag and builds the model without opening a browser.
    empty = os.path.join(work, "empty.jsonl")
    open(empty, "w").close()
    sys.argv = ["run.py", "--test_file", empty, "--model_backend", "vllm", "--model_name", "mock-model",
                "--vllm_base_url", base_url, "--output_dir", os.path.join(work, "results"),
                "--run_subdir", "smoke", "--headless", "--max_task_attempts", "2",
                "--chrome_profile_dir", os.path.join(work, "profile")]
    wv_run.main()
    assert os.path.isdir(os.path.join(work, "results", "smoke")), "run_subdir not used"
    print("run.py main (flags, model, Chrome options): OK")

    # The agent's model call, with an OpenAI-format image message.
    from benchmark_adapters.webvoyager_adapter import build_model_from_args
    args = argparse.Namespace(model_name="mock-model", model_backend="vllm", vllm_base_url=base_url,
                              temperature=0.5)
    model = build_model_from_args(args=args)
    png = os.path.join(work, "shot.png")
    Image.new("RGB", (8, 8)).save(png)
    from utils import encode_image
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(png)}"}}]}]
    n = len(requests)
    reply, prompt_tokens, completion_tokens, error = wv_run.call_model(args, model, messages)
    body = requests[n]["body"]
    assert not error and reply.startswith("Thought:"), reply
    assert (prompt_tokens, completion_tokens) == (11, 7), (prompt_tokens, completion_tokens)
    assert body["temperature"] == 0.5, body.get("temperature")
    assert body["messages"][1]["content"][1]["type"] == "image_url", body["messages"][1]
    print(f"run.py call_model: OK  temperature={body['temperature']}  tokens=({prompt_tokens}, {completion_tokens})")

    # PDF answering: the PDF's text must reach the model.
    from benchmark_adapters.webvoyager_adapter import answer_from_pdf
    pdf = os.path.join(work, "doc.pdf")
    write_pdf(path=pdf, text="The answer is forty two")
    n = len(requests)
    answer = answer_from_pdf(model=model, pdf_path=pdf, question="What is the answer?")
    sent = requests[n]["body"]["messages"][1]["content"]
    assert "The answer is forty two" in sent and "What is the answer?" in sent, sent
    assert answer.startswith("Thought:"), answer
    print("answer_from_pdf: OK  (PDF text and question reached the model)")


def check_auto_eval(*, base_url, work):
    results = os.path.join(work, "process")
    done = os.path.join(results, "taskAllrecipes--0")
    os.makedirs(done)
    with open(os.path.join(done, "interact_messages.json"), "w") as f:
        json.dump([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "Now given a task: find a recipe  Please interact with https://x"}]},
            {"role": "assistant", "content": "Thought: found it.\nAction: ANSWER; pancakes"},
        ], f)
    Image.new("RGB", (8, 8)).save(os.path.join(done, "screenshot1.png"))
    errored = os.path.join(results, "taskAmazon--0")
    os.makedirs(errored)
    with open(os.path.join(errored, "task_error.json"), "w") as f:
        json.dump({"error": "browser died"}, f)
    os.makedirs(os.path.join(results, "taskApple--0"))  # started, never finished

    proc = subprocess.run(
        [sys.executable, "auto_eval.py", "--model_backend", "vllm", "--model_name", "mock-judge",
         "--vllm_base_url", base_url, "--process_dir", results, "--max_attached_imgs", "1"],
        cwd=os.path.join(WEBVOYAGER, "evaluation"), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with open(os.path.join(results, "scores.json")) as f:
        scores = json.load(f)
    expected = {"taskAllrecipes--0": "success", "taskAmazon--0": "browser_error", "taskApple--0": "not_run"}
    got = {"task" + k: v for k, v in scores["tasks"].items()}
    assert got == expected, got
    print(f"auto_eval: OK  success={scores['success']} browser_error={scores['browser_error']} "
          f"not_run={scores['not_run']}")


def main():
    # The judge reads "SUCCESS"; the agent reads Thought/Action. One reply serves both.
    server, base_url, requests = start_mock_vllm(reply="Thought: done.\nAction: ANSWER; SUCCESS")
    work = tempfile.mkdtemp()
    check_run_py(base_url=base_url, requests=requests, work=work)
    check_auto_eval(base_url=base_url, work=work)
    server.shutdown()
    print("ALL OK")


if __name__ == "__main__":
    main()
