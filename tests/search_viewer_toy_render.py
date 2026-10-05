"""
TOY (skill_discovery.md §1.5; delete with the rest of the toy scaffolding): render a toy search
viewer page in headless Chromium and check that its script ran (tree nodes, detail pane, chart,
no JS errors); saves a screenshot next to it. Inside the container:

    bash scripts/container.sh python tests/search_viewer_toy_render.py --run_dir <out dir>
"""
import os
import click


@click.command()
@click.option("--run_dir", required=True)
def main(run_dir):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    opts = webdriver.ChromeOptions()
    opts.binary_location = os.environ["CUSI_CHROME_BINARY"]
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1400,1800"):
        opts.add_argument(a)
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(service=Service(os.environ["CUSI_CHROMEDRIVER"]), options=opts)
    try:
        driver.get("file://" + os.path.abspath(os.path.join(run_dir, "index.html")))
        n_nodes = driver.execute_script("return document.querySelectorAll('.n').length")
        detail = driver.execute_script("return document.getElementById('detail').innerText.slice(0, 200)")
        chart = driver.execute_script("return document.querySelectorAll('#chart polyline').length")
        rows = driver.execute_script("return document.querySelectorAll('#sel tr').length")
        broken = driver.execute_script("return [...document.images].filter(i => i.complete && i.naturalWidth === 0).length")
        errors = [l for l in driver.get_log("browser") if l["level"] == "SEVERE"]
        driver.save_screenshot(os.path.join(run_dir, "viewer_screenshot.png"))
        print(f"tree nodes {n_nodes}, chart lines {chart}, selection rows {rows}, broken images {broken}, "
              f"JS errors {len(errors)}")
        print("detail:", detail.replace("\n", " | "))
        for e in errors[:5]:
            print("  ", e["message"][:200])
        ok = n_nodes > 0 and chart == 4 and rows > 1 and not errors
        print("ALL OK" if ok else "FAIL")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
