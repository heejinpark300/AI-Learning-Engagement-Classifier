
from pathlib import Path
from collections import defaultdict
import hashlib

BASE = Path(
    r"C:\PHJ_project_20261019\DeepLearning_Project_1102"
)

DATA = BASE / "Final Dataset 256"

groups = defaultdict(list)

for image in DATA.rglob("*"):
    if image.suffix.lower() not in [
        ".jpg", ".jpeg", ".png", ".webp"
    ]:
        continue

    with open(image, "rb") as f:
        image_hash = hashlib.sha256(f.read()).hexdigest()

    groups[image_hash].append(str(image.relative_to(DATA)))

duplicate_groups = {
    key: files
    for key, files in groups.items()
    if len(files) > 1
}

print("전체 이미지:", sum(len(v) for v in groups.values()))
print("완전히 동일한 이미지 그룹:", len(duplicate_groups))
print(
    "중복 파일 수:",
    sum(len(v) - 1 for v in duplicate_groups.values())
)

for files in list(duplicate_groups.values())[:10]:
    print("\n중복 이미지:")
    for file in files:
        print(" ", file)

