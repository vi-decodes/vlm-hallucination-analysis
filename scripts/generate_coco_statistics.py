
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# Input and output paths
INPUT_FILE = Path("annotations/instances_val2014.json")
OUTPUT_DIR = Path("output/coco")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load COCO annotations
print("Loading COCO validation annotations...")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    coco = json.load(f)

# Map category IDs to category names
categories = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

# Map each image to its unique object categories
image_objects = defaultdict(set)

for annotation in coco["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    image_objects[image_id].add(categories[category_id])

print("Total validation images:", len(coco["images"]))
print("Images with annotations:", len(image_objects))

# 1. Category frequency
# Number of images containing each category
category_freq = defaultdict(int)

for objects in image_objects.values():
    for obj in objects:
        category_freq[obj] += 1

# 2. Co-occurrence frequency
# Number of images containing each pair of categories
co_occurrence = defaultdict(lambda: defaultdict(int))

for objects in image_objects.values():
    for obj1, obj2 in combinations(sorted(objects), 2):
        co_occurrence[obj1][obj2] += 1
        co_occurrence[obj2][obj1] += 1

# 3. Co-occurrence probability
# P(obj2 | obj1) = count(obj1 and obj2) / count(obj1)
co_occurrence_probability = {}

for obj1, related_objects in co_occurrence.items():
    co_occurrence_probability[obj1] = {
        obj2: round(count / category_freq[obj1], 6)
        for obj2, count in related_objects.items()
    }

# Save category frequencies
with (OUTPUT_DIR / "category_freq.json").open("w") as f:
    json.dump(dict(category_freq), f, indent=2)

# Save co-occurrence counts
with (OUTPUT_DIR / "co_occurrence.json").open("w") as f:
    json.dump(
        {obj: dict(counts) for obj, counts in co_occurrence.items()},
        f,
        indent=2
    )

# Save conditional probabilities
with (OUTPUT_DIR / "co_occurrence_probability.json").open("w") as f:
    json.dump(co_occurrence_probability, f, indent=2)

print("\nStatistics generated successfully!")
print("Unique categories:", len(category_freq))
print("Category frequency:", OUTPUT_DIR / "category_freq.json")
print("Co-occurrence counts:", OUTPUT_DIR / "co_occurrence.json")
print("Co-occurrence probabilities:",
      OUTPUT_DIR / "co_occurrence_probability.json")

# Show the 10 most frequent categories
print("\nTop 10 categories:")
for obj, count in sorted(
    category_freq.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:
    print(f"{obj}: {count}")
