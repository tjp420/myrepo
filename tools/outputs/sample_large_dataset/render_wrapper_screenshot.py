from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent
HTML = OUT_DIR / "histogram_thumbnails_wrapper.html"
OUT_PNG = OUT_DIR / "histogram_thumbnails_wrapper.png"


def main():
    if not HTML.exists():
        raise SystemExit(f"Missing HTML: {HTML}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 600})
        page.goto(HTML.as_uri())
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT_PNG), full_page=True)
        browser.close()

    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()
