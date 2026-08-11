import html
import json
import re
import time
from pathlib import Path

import requests

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "GestureMedicalImaging/1.0 (educational demo project)"}
ALLOWED_LICENSES = {"cc0", "pd", "public domain", "cc-by", "cc-by-sa", "cc-by-2.0",
                     "cc-by-3.0", "cc-by-4.0", "cc-by-sa-2.0", "cc-by-sa-3.0", "cc-by-sa-4.0"}

# Titles containing these are excluded: old journal scans, annotated teaching
# diagrams, and non-frontal views the model wasn't trained to interpret.
EXCLUDED_TITLE_WORDS = ["journal", "roentgenology", "lateral", "annotated", "labeled",
                        "labelled", "diagram", "illustration", "structures", "anatomy",
                        "drawing", "schematic", "textbook"]

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

# output filename -> list of search terms to try in order
TARGETS = {
    "normal_1.jpg": ["normal chest radiograph PA"],
    "normal_2.jpg": ["normal chest x-ray healthy"],
    "pneumonia.jpg": ["pneumonia chest radiograph consolidation"],
    "cardiomegaly.jpg": ["cardiomegaly posteroanterior chest radiograph",
                          "enlarged heart chest x-ray frontal", "cardiomegaly PA chest x-ray"],
    "pleural_effusion.jpg": ["pleural effusion chest radiograph"],
    "pneumothorax.jpg": ["pneumothorax posteroanterior chest radiograph",
                          "pneumothorax frontal chest x-ray", "collapsed lung PA radiograph"],
    "nodule.jpg": ["pulmonary nodule chest radiograph", "solitary pulmonary nodule x-ray"],
    "lung_mass.jpg": ["lung mass chest radiograph tumor", "lung cancer chest x-ray mass"],
}
ATTRIBUTIONS_JSON = "attribution.json"

MIN_WIDTH = 400
MIN_HEIGHT = 400


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return html.unescape(text).strip()


def api_get(params: dict, retries: int = 6):
    delay = 5
    for attempt in range(retries):
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
        time.sleep(1)
        return resp
    resp.raise_for_status()
    return resp


def search_candidates(term: str, limit: int = 10):
    resp = api_get({
        "action": "query", "list": "search", "srnamespace": 6,
        "srsearch": term, "srlimit": limit, "format": "json",
    })
    results = resp.json().get("query", {}).get("search", [])
    titles = [r["title"] for r in results if re.search(r"\.(jpe?g|png)$", r["title"], re.I)]
    return titles


def get_imageinfo(title: str):
    resp = api_get({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "format": "json",
    })
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo")
        if info:
            return info[0]
    return None


def pick_licensed_image(term: str):
    for title in search_candidates(term):
        lowered = title.lower()
        if any(word in lowered for word in EXCLUDED_TITLE_WORDS):
            continue
        info = get_imageinfo(title)
        if not info:
            continue
        if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
            continue
        meta = info.get("extmetadata", {})
        license_short = meta.get("LicenseShortName", {}).get("value", "").lower()
        if not any(allowed in license_short for allowed in ALLOWED_LICENSES):
            continue
        return {
            "title": title,
            "url": info["url"],
            "width": info["width"],
            "height": info["height"],
            "license": meta.get("LicenseShortName", {}).get("value", "Unknown"),
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "artist": strip_html(meta.get("Artist", {}).get("value", "Unknown")),
            "source": info.get("descriptionurl", ""),
        }
    return None


def download(url: str, dest: Path):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def load_attributions() -> dict:
    path = IMAGES_DIR / ATTRIBUTIONS_JSON
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_attributions(attributions: dict):
    (IMAGES_DIR / ATTRIBUTIONS_JSON).write_text(
        json.dumps(attributions, indent=2), encoding="utf-8")

    lines = ["# Image Attribution\n",
             "Sample chest radiographs fetched from Wikimedia Commons, openly licensed.\n"]
    for filename, info in sorted(attributions.items()):
        lines.append(
            f"- **{filename}** — \"{info['title']}\" by {info['artist']}, "
            f"licensed {info['license']} ({info['license_url']}). Source: {info['source']}"
        )
    (IMAGES_DIR / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    attributions = load_attributions()
    fetched_now = 0

    for filename, terms in TARGETS.items():
        if (IMAGES_DIR / filename).exists():
            print(f"Skipping {filename}, already downloaded")
            continue

        picked = None
        for term in terms:
            print(f"Searching: {term} ...")
            picked = pick_licensed_image(term)
            if picked:
                break
            time.sleep(1)

        if not picked:
            print(f"  no openly-licensed match found for {filename}, skipping")
            continue

        dest = IMAGES_DIR / filename
        download(picked["url"], dest)
        print(f"  saved {filename} ({picked['width']}x{picked['height']}, {picked['license']})")

        attributions[filename] = picked
        fetched_now += 1
        time.sleep(0.5)

    save_attributions(attributions)
    print(f"\nDone. {fetched_now} new image(s) fetched this run, "
          f"{len(attributions)}/{len(TARGETS)} total available.")


if __name__ == "__main__":
    main()
