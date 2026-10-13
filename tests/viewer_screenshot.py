"""
Render a run viewer (cusi/search/viewer.py) in headless Chrome and screenshot its views, printing any JS errors.
Run inside the CUSI container (it has Chrome; CUSI_CHROME_BINARY / CUSI_CHROMEDRIVER):
    bash scripts/container.sh python -m tests.viewer_screenshot --index <run dir>/viewer/index.html --out_dir <dir>
"""
import os
import time
import click


@click.command()
@click.option("--index", required=True, help="The viewer's index.html.")
@click.option("--out_dir", required=True)
def main(index, out_dir):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    os.makedirs(out_dir, exist_ok=True)
    opts = webdriver.ChromeOptions()
    for a in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=1500,950",
              "--allow-file-access-from-files"):
        opts.add_argument(a)
    if os.environ.get("CUSI_CHROME_BINARY"):
        opts.binary_location = os.environ["CUSI_CHROME_BINARY"]
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    service = Service(executable_path=os.environ["CUSI_CHROMEDRIVER"]) if os.environ.get("CUSI_CHROMEDRIVER") else None
    driver = webdriver.Chrome(options=opts, service=service)
    try:
        url = "file://" + os.path.abspath(index)
        views = [("tree", ""), ("node", "#n=3"), ("expansion", "#e=2&s=4"), ("expansion_start", "#e=0&s=0")]
        for name, frag in views:
            driver.get("about:blank")
            driver.get(url + frag)
            time.sleep(1.5)
            driver.save_screenshot(os.path.join(out_dir, f"{name}.png"))
        driver.execute_script("document.querySelector('nav button[data-tab=timeTab]').click()")
        time.sleep(0.8)
        driver.save_screenshot(os.path.join(out_dir, "timeline.png"))
        driver.execute_script("document.querySelector('nav button[data-tab=expTab]').click()")
        time.sleep(0.5)
        driver.save_screenshot(os.path.join(out_dir, "expansions.png"))
        driver.execute_script("document.documentElement.dataset.theme='dark';"
                              "document.querySelector('nav button[data-tab=treeTab]').click()")
        driver.get(url + "#e=1&s=2")
        driver.execute_script("document.documentElement.dataset.theme='dark'")
        time.sleep(1.0)
        driver.save_screenshot(os.path.join(out_dir, "dark.png"))
        errors = [e for e in driver.get_log("browser") if e["level"] in ("SEVERE", "WARNING")]
        print("JS errors:", errors or "none")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
