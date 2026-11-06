
from pathlib import Path
import csv
import hashlib
import shutil
from collections import Counter

BASE = Path(
    r"C:\PHJ_project_20261019\DeepLearning_Project_1102"
)

SOURCE = BASE / "dataset" / "unique_images"
TARGET = BASE / "dataset" / "clean_binary"
REPORT = BASE / "results" / "duplicate_report.csv"

# 충돌 원본 이미지의 SHA256 계산
conflict_hashes = set()

with REPORT.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    for row in csv.DictReader(f):
        original_path = Path(row["original_path"])
        if original_path.exists():
            conflict_hashes.add(
                hashlib.sha256(
                    original_path.read_bytes()
                ).hexdigest()
            )

counts = Counter()

for cls in ["Engagement", "Disengagement"]:
    output_dir = TARGET / cls
    output_dir.mkdir(parents=True, exist_ok=True)

    for image in (SOURCE / cls).iterdir():
        if not image.is_file():
            continue

        image_hash = hashlib.sha256(
            image.read_bytes()
        ).hexdigest()

        if image_hash in conflict_hashes:
            counts["excluded"] += 1
            continue

        shutil.copy2(
            image,
            output_dir / image.name
        )

        counts[cls] += 1

print("\n=== 라벨 충돌 제거 완료 ===")
print("Engagement:", counts["Engagement"])
print("Disengagement:", counts["Disengagement"])
print("충돌 제외:", counts["excluded"])
print("최종 이미지:", counts["Engagement"] + counts["Disengagement"])
print("저장 위치:", TARGET)
