
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import shutil

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "dataset" / "clean_binary"
TARGET = BASE / "dataset" / "final_split"

CLASSES = ["Engagement", "Disengagement"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 64비트 dHash: 비슷한 이미지 그룹 찾기
def dhash(path):
    with Image.open(path) as img:
        img = img.convert("L").resize((9, 8))
        pixels = np.asarray(img)

    bits = pixels[:, 1:] > pixels[:, :-1]
    result = 0
    for bit in bits.flatten():
        result = (result << 1) | int(bit)
    return result


def hamming(a, b):
    return (a ^ b).bit_count()


files = []
labels = []
hashes = []

for cls in CLASSES:
    for path in sorted((SOURCE / cls).iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        files.append(path)
        labels.append(CLASSES.index(cls))
        hashes.append(dhash(path))

print("분석 이미지:", len(files), flush=True)

# 유사 이미지 연결을 위한 Union-Find
parent = list(range(len(files)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


# 64비트를 8개 구간으로 나눠 후보 탐색
buckets = defaultdict(list)

for i, value in enumerate(hashes):
    candidates = set()

    for band in range(8):
        key = (band, (value >> (band * 8)) & 255)
        candidates.update(buckets[key])

    for j in candidates:
        if hamming(value, hashes[j]) <= 4:
            union(i, j)

    for band in range(8):
        key = (band, (value >> (band * 8)) & 255)
        buckets[key].append(i)

groups = [find(i) for i in range(len(files))]

# 서로 다른 클래스가 한 유사 그룹에 존재하면 보류
group_labels = defaultdict(set)
for group, label in zip(groups, labels):
    group_labels[group].add(label)

valid_indices = [
    i for i, group in enumerate(groups)
    if len(group_labels[group]) == 1
]

excluded = len(files) - len(valid_indices)
print("라벨 충돌 유사 그룹 제외 이미지:", excluded)

files = [files[i] for i in valid_indices]
labels = np.array([labels[i] for i in valid_indices])
groups = np.array([groups[i] for i in valid_indices])

if len(set(labels)) != 2 or len(set(groups)) < 10:
    raise RuntimeError("데이터 그룹 수를 확인하세요.")

# 그룹 단위로 10개 폴드 생성
splitter = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

fold_ids = np.full(len(files), -1)

for fold, (_, test_index) in enumerate(
    splitter.split(
        np.zeros(len(files)),
        labels,
        groups
    )
):
    fold_ids[test_index] = fold

if np.any(fold_ids < 0):
    raise RuntimeError("데이터 분리 오류")

# 마지막 2개 폴드를 검증/시험용으로 사용
split_names = {
    i: "train" if i < 8 else
       ("val" if i == 8 else "test")
    for i in range(10)
}

if TARGET.exists():
    raise RuntimeError(
        "final_split 폴더가 이미 있습니다. "
        "기존 결과를 먼저 확인하세요."
    )

counts = Counter()

for path, label, fold in zip(files, labels, fold_ids):
    split = split_names[int(fold)]
    cls = CLASSES[int(label)]

    destination = TARGET / split / cls
    destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(path, destination / path.name)
    counts[(split, cls)] += 1

print("\n=== 데이터 분리 완료 ===")

for split in ["train", "val", "test"]:
    print(f"\n[{split}]")
    for cls in CLASSES:
        print(cls, counts[(split, cls)])

print("\n저장 위치:", TARGET)
