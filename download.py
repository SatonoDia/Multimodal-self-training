import os
import time
import requests
import tqdm
from datasets import load_dataset

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, dst_path, session=None, retry=3, timeout=30):
    session = session or requests.Session()
    for attempt in range(retry):
        try:
            r = session.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"Download error ({attempt+1}/{retry}) for {url}: {e}")
            time.sleep(2)
    return False

def main():
    hard_dir = "/root/autodl-tmp/data/mathv360k/hard"
    ensure_dir(hard_dir)

    ds = load_dataset("Zhiqiang007/MathV360K", split="train") 

    print("Columns available:", ds.column_names)

    session = requests.Session()

    for idx, item in enumerate(tqdm.tqdm(ds)):
        img_url = item.get("image") or item.get("image_url") or item.get("image_path")
        complexity = item.get("image_complexity")

        if img_url is None:
            continue
        if complexity not in (2, 3):
            continue

        fname = os.path.basename(img_url)
        dst = os.path.join(hard_dir, fname)

        if os.path.exists(dst):
            continue

        success = download_file(img_url, dst, session=session)
        if not success:
            print(f"Failed to download: {img_url}")

    print("Done downloading hard‐group images.")

if __name__ == "__main__":
    main()
