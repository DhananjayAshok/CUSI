"""Part 0 verification: one text and one image request to the served model through
cusi.utils (build_model(model_backend="vllm")), plus a rough throughput measurement.

    python tests/vllm_smoke.py --model_name <served name>    (with the server from scripts/serve_vllm.sh up)
"""
import time
import click
import numpy as np
from PIL import Image
from cusi.utils import load_parameters, build_model, log_info

parameters = load_parameters()


@click.command()
@click.option("--model_name", required=True, help="Model name the vLLM server serves.")
def main(model_name):
    model = build_model(model_name=model_name, model_backend="vllm", parameters=parameters)

    out = model.infer(texts="What is the capital of France? Answer in one word.", max_new_tokens=20)
    log_info(f"text: {out['output']!r} meta={out['meta']}", parameters=parameters)
    assert "paris" in out["output"].lower(), out

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :160] = (255, 0, 0)
    img[:, 160:] = (0, 0, 255)
    out = model.infer(texts="What two colours does this image show? Answer with two words.",
                      images=[Image.fromarray(img)], max_new_tokens=20)
    log_info(f"image: {out['output']!r} meta={out['meta']}", parameters=parameters)
    assert "red" in out["output"].lower() and "blue" in out["output"].lower(), out

    for batch in (1, 16, 32):
        t = time.time()
        out = model.infer(texts=["Write a 100-word story about a robot."] * batch, max_new_tokens=150)
        dt = time.time() - t
        n_tok = sum(out["meta"]["output_tokens"])
        log_info(f"throughput: batch {batch}: {n_tok} output tokens in {dt:.1f}s = {n_tok / dt:.0f} tok/s",
                 parameters=parameters)
    t = time.time()
    out = model.infer(texts=["Describe this image in one sentence."] * 16, images=[[Image.fromarray(img)]] * 16,
                      max_new_tokens=60)
    log_info(f"throughput: 16 image requests in {time.time() - t:.1f}s", parameters=parameters)
    print("VLLM SMOKE OK")


if __name__ == "__main__":
    main()
