import json
import numpy as np
from pathlib import Path

from src.face_feature import get_face_feature


FACE_LIB_PATH = Path("/home/chen/yolo/face_lib.json")


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.dot(a, b)
        /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


def load_face_lib():
    with open(
        FACE_LIB_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def recognize(img_path):

    face_lib = load_face_lib()

    query_embedding = get_face_feature(img_path)

    best_name = "unknown"
    best_score = -1

    for name, feature in face_lib.items():

        score = cosine_similarity(
            query_embedding,
            feature
        )

        if score > best_score:
            best_score = score
            best_name = name

    threshold = 0.45

    if best_score < threshold:
        best_name = "unknown"

    return {
        "name": best_name,
        "similarity": best_score,
        "status": "known"
        if best_name != "unknown"
        else "stranger"
    }


if __name__ == "__main__":

    test_image = "/home/chen/yolo/test_data/崔希民.jpg"

    result = recognize(test_image)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )
