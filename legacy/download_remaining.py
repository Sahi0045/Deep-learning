"""
Quick downloader for remaining 6 classes — runs 2 fast queries per class.
"""
import os, shutil, glob, logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REMAINING = {
    "quarrel":     ["people arguing quarrel angry confrontation photo", "angry argument two people shouting"],
    "isolation":   ["child excluded alone bullying school photo", "student sitting alone isolated group"],
    "laughing":    ["people mocking laughing bullying taunting photo", "group laughing pointing person mockery"],
    "gossiping":   ["people whispering gossiping secrets photo", "students gossiping bullying whispering"],
    "pullinghair": ["person pulling hair fight real photo", "hair grabbing fight assault photo"],
    "stabbing":    ["knife threat confrontation aggressive photo", "weapon threat person aggressive real"],
}

BASE = "training_data"

def is_valid(path):
    try:
        img = cv2.imread(path)
        if img is None: return False
        h,w = img.shape[:2]
        return h >= 80 and w >= 80 and len(img.shape) >= 3
    except: return False

def download_class(cls, queries, num=70):
    from icrawler.builtin import BingImageCrawler
    out_dir = os.path.join(BASE, cls)
    # Remove synthetic images
    for f in glob.glob(os.path.join(out_dir, "img_*.jpg")):
        os.remove(f)
    
    collected = 0
    for qi, q in enumerate(queries):
        tmp = os.path.join(out_dir, f"_tmp{qi}")
        os.makedirs(tmp, exist_ok=True)
        try:
            c = BingImageCrawler(feeder_threads=1, parser_threads=2,
                                 downloader_threads=6,
                                 storage={"root_dir": tmp},
                                 log_level=logging.ERROR)
            c.crawl(keyword=q, max_num=num, filters=dict(type="photo"))
        except Exception as e:
            print(f"  [{cls}] query {qi} error: {e}")
        
        for f in glob.glob(os.path.join(tmp, "*")):
            if is_valid(f):
                ext = Path(f).suffix or ".jpg"
                dst = os.path.join(out_dir, f"real_{collected:05d}{ext}")
                shutil.move(f, dst)
                collected += 1
        shutil.rmtree(tmp, ignore_errors=True)
    
    print(f"  [{cls}] Downloaded {collected} valid images")
    return collected

print("Downloading remaining 6 classes...")
total = 0
for cls, queries in REMAINING.items():
    print(f"\n▶ {cls}:")
    total += download_class(cls, queries, num=70)

print(f"\nDone! Total new images: {total}")
