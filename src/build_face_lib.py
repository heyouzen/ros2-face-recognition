import os
import numpy as np
import json
from face_feature import get_face_feature

FACE_DB_ROOT = "/home/chen/yolo/face_database"
SAVE_LIB_PATH = "/home/chen/yolo/face_lib.json"

def merge_features(feat_list):
    return np.mean(np.array(feat_list), axis=0)

def build_single_person(person_dir):
    feat_list = []
    for fname in os.listdir(person_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(person_dir, fname)
            try:
                feat = get_face_feature(img_path)
                feat_list.append(feat)
                print(f"已读取图片：{fname}")
            except Exception as e:
                print(f"跳过无效图片 {fname}，错误：{e}")
    if len(feat_list) == 0:
        return None
    merge_feat = merge_features(feat_list)
    return merge_feat.tolist()

def build_all_lib():
    face_lib = {}
    for name_dir in os.listdir(FACE_DB_ROOT):
        person_path = os.path.join(FACE_DB_ROOT, name_dir)
        if not os.path.isdir(person_path):
            continue
        print(f"\n==== 正在处理人员：{name_dir} ====")
        merge_feat = build_single_person(person_path)
        if merge_feat is not None:
            face_lib[name_dir] = merge_feat
    with open(SAVE_LIB_PATH, "w", encoding="utf-8") as f:
        json.dump(face_lib, f, ensure_ascii=False, indent=2)
    print("✅ 所有人脸特征融合完成，已生成 face_lib.json")

if __name__ == "__main__":
    build_all_lib()
