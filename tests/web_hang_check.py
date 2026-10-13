"""
A frozen page (a busy main thread) must make a WebDriver call raise within the client timeout instead of hanging, as
cusi/envs/webvoyager.py relies on (WEBDRIVER_TIMEOUT, WebDriverHang). Run in the CUSI container:
    bash scripts/container.sh python -m tests.web_hang_check
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.remote_connection import RemoteConnection
from cusi.envs.webvoyager import WebDriverHang

opts = webdriver.ChromeOptions()
for a in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage"):
    opts.add_argument(a)
if os.environ.get("CUSI_CHROME_BINARY"):
    opts.binary_location = os.environ["CUSI_CHROME_BINARY"]
service = Service(executable_path=os.environ["CUSI_CHROMEDRIVER"]) if os.environ.get("CUSI_CHROMEDRIVER") else None
RemoteConnection.set_timeout(15)
try:
    driver = webdriver.Chrome(options=opts, service=service)
finally:
    RemoteConnection.reset_timeout()
driver.get("data:text/html,<p>ok</p>")
driver.execute_script("setTimeout(() => { while (true) {} }, 10)")
time.sleep(1)
t0 = time.time()
try:
    driver.execute_script("return document.title")
    print("FAIL: the call returned on a frozen page")
except WebDriverHang as e:
    print(f"ok    a frozen page raises {type(e).__name__} after {time.time() - t0:.0f}s (client timeout 15s)")
except Exception as e:
    print(f"other {type(e).__name__} after {time.time() - t0:.0f}s: {str(e)[:200]}")
try:
    driver.quit()
except Exception as e:
    print(f"quit raised {type(e).__name__} (ignored, as _quit does)")
