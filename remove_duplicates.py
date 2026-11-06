
from pathlib import Path
from collections import Counter
import hashlib
import shutil
import csv

BASE = Path(
    r"C:\PHJ_project_20261019\DeepLearning_Project_1102"
)

SOURCE = BASE / "Final Dataset 256"
TARGET = BASE / "dataset" / "unique_images"
REPORT = BASE / "results" / "duplicate_report.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

hash_records = {}
counts = Counter()
conflicts = []

# 첫 실행에서만 새 출력 폴더를 사용
if TARGET.exists() and any(TARGET.rglob("*.*")):
    raise RuntimeError(
        "출력 폴더에 파일이 있습니다. "
        "기존 결과를 확인한 뒤 진행하세요."
    )

for cls in ["Engagement", "Disengagement"]:
    source_folder = SOURCE / cls
    target_folder = TARGET / cls
    target_folder.mkdir(parents=True, exist_ok=True)

    for path in sorted(source_folder.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        image_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if image_hash in hash_records:
            original_cls, original_path = hash_records[image_hash]

            if original_cls != cls:
                conflicts.append([
                    str(original_path),
                    str(path),
                    original_cls,
                    cls
                ])

            counts["duplicates"] += 1
            continue

        hash_records[image_hash] = (cls, path)

        # 원본 파일명 충돌 방지
        new_name = image_hash[:16] + path.suffix.lower()
        destination = target_folder / new_name

        shutil.copy2(path, destination)

        counts[cls] += 1

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "original_path",
        "duplicate_path",
        "original_class",
        "duplicate_class"
    ])
    writer.writerows(conflicts)

print("\n=== 중복 제거 완료 ===")
print("Engagement:", counts["Engagement"])
print("Disengagement:", counts["Disengagement"])
print("중복 제거:", counts["duplicates"])
print("라벨 충돌:", len(conflicts))
print("고유 이미지:", counts["Engagement"] + counts["Disengagement"])
print("저장 위치:", TARGET)
