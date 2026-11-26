"""피트니스 데이터 무결성 검사(읽기 전용).

- 원본 폴더(data/raw)는 읽기만 한다. 결과는 표준 출력으로만 낸다.
- Test 분할은 무결성(파일·수치·겹침·해상도·디코딩·중복)만 확인한다. 분포 EDA는 Train만 한다.

실행: uv run python checks/check_data_integrity.py data/raw
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

OFFICIAL = {  # 주제 안내문 표(이미지, 사람 코드, 클립)
    "train": (8144, 16, 1018),
    "val": (1440, 3, 180),
    "test": (1432, 3, 179),
}


def section(title: str) -> None:
    print(f"\n## {title}")


def main(raw: Path) -> None:
    md = pd.read_csv(raw / "metadata.csv", encoding="utf-8-sig")
    print(f"metadata.csv rows={len(md)} cols={list(md.columns)}")

    section("1. 행·식별자")
    print("image_id unique:", md["image_id"].is_unique, "| image_path unique:", md["image_path"].is_unique)
    print("null counts (nonzero):", {k: int(v) for k, v in md.isna().sum().items() if v})
    print("split values:", dict(Counter(md["split"])))
    print("path prefix == images/<split>/:",
          bool((md["image_path"].str.split("/").str[1] == md["split"]).all()))

    section("2. 분할별 수치 vs 공식 표")
    for sp, (n_img, n_p, n_c) in OFFICIAL.items():
        d = md[md["split"] == sp]
        got = (len(d), d["person_id"].nunique(), d["clip_id"].nunique())
        print(f"{sp:5s} got={got} official={(n_img, n_p, n_c)} match={got == (n_img, n_p, n_c)}")
    print("total classes:", md["target_id"].nunique(), "| total persons:", md["person_id"].nunique(),
          "| total clips:", md["clip_id"].nunique())
    print("classes per split:", {sp: md.loc[md.split == sp, "target_id"].nunique() for sp in OFFICIAL})

    section("3. 분할 간 겹침(누수)")
    for col in ["person_id", "source_subject_id", "clip_id", "image_id"]:
        sets = {sp: set(md.loc[md.split == sp, col]) for sp in OFFICIAL}
        print(f"{col:18s} train∩val={len(sets['train'] & sets['val'])} "
              f"train∩test={len(sets['train'] & sets['test'])} val∩test={len(sets['val'] & sets['test'])}")
    print("person_id == source_subject_id (all rows):", bool((md["person_id"] == md["source_subject_id"]).all()))
    pid_src = md.groupby("person_id")["source_subject_id"].nunique()
    print("person_id with >1 source_subject_id:", int((pid_src > 1).sum()))

    section("4. 클립 구성")
    g = md.groupby("clip_id")
    per_clip = g.size()
    print("images per clip:", dict(Counter(per_clip)))
    for col in ["person_id", "target_id", "split", "camera", "state_code"]:
        multi = int((g[col].nunique() > 1).sum())
        print(f"clips with >1 {col}: {multi}")
    dup_frames = int(md.duplicated(["clip_id", "frame_no"]).sum())
    print("duplicate (clip_id, frame_no):", dup_frames)
    print("duplicate (clip_id, camera, frame_no):", int(md.duplicated(["clip_id", "camera", "frame_no"]).sum()))
    shape = g.agg(n_cam=("camera", "nunique"), n_frame=("frame_no", "nunique"))
    print("clip shape (n_camera, n_frame_no) counts:", dict(Counter(zip(shape.n_cam, shape.n_frame))))
    ex = md[md.clip_id == md.clip_id.iloc[0]].sort_values(["camera", "frame_no"])
    print("example clip (camera, frame_no):", list(zip(ex.camera, ex.frame_no)))
    # 같은 사람·운동·state_code 안에서 여러 클립이 있는지(클립 간 유사성 위험)
    k = md.groupby(["person_id", "target_id", "state_code"])["clip_id"].nunique()
    print("clips per (person, target, state_code):", dict(Counter(k)))
    print("frame_no range:", int(md.frame_no.min()), "-", int(md.frame_no.max()),
          "| state_code unique:", md.state_code.nunique())

    section("5. 파일 대응·해상도·디코딩·해시 중복")
    files_on_disk = {p.relative_to(raw).as_posix() for p in (raw / "images").rglob("*") if p.is_file()}
    listed = set(md["image_path"])
    print("files on disk:", len(files_on_disk), "| missing:", len(listed - files_on_disk),
          "| extra:", len(files_on_disk - listed))
    print("metadata width×height:", dict(Counter(zip(md["width"], md["height"]))))
    bad_decode, size_mismatch, dim_mismatch, mode_counter = [], 0, 0, Counter()
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    actual_dims = Counter()
    for row in md.itertuples(index=False):
        p = raw / row.image_path
        b = p.read_bytes()
        if len(b) != row.file_size_bytes:
            size_mismatch += 1
        hashes[hashlib.sha256(b).hexdigest()].append((row.split, row.clip_id))
        try:
            with Image.open(p) as im:
                im.load()
                actual_dims[im.size] += 1
                mode_counter[im.mode] += 1
                if im.size != (row.width, row.height):
                    dim_mismatch += 1
        except Exception as e:  # noqa: BLE001
            bad_decode.append((row.image_path, repr(e)[:80]))
    print("actual dims:", dict(actual_dims), "| modes:", dict(mode_counter))
    print("decode failures:", len(bad_decode), bad_decode[:5])
    print("file_size mismatch:", size_mismatch, "| dim mismatch vs metadata:", dim_mismatch)
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    cross_split = [v for v in dup_groups if len({s for s, _ in v}) > 1]
    cross_clip = [v for v in dup_groups if len({c for _, c in v}) > 1]
    print("exact-duplicate groups:", len(dup_groups), "| images in them:", sum(len(v) for v in dup_groups),
          "| groups spanning splits:", len(cross_split), "| groups spanning clips:", len(cross_clip))

    section("6. Train 부분집합 CSV(25/50/100%)")
    tr_ids = set(md.loc[md.split == "train", "image_id"])
    subs = {}
    vt = md[md.split != "train"]
    for pct in (25, 50, 100):
        s = pd.read_csv(raw / f"metadata_train_{pct}pct.csv", encoding="utf-8-sig")
        subs[pct] = s
        st = s[s.split == "train"]
        sv = s[s.split != "train"]
        full_clip = md[md.clip_id.isin(st.clip_id)].groupby("clip_id").size()
        whole_clips = bool((st.groupby("clip_id").size() == full_clip).all())
        print(f"{pct:3d}%: rows={len(s)} splits={dict(Counter(s['split']))} "
              f"train_part_subset_of_train={set(st.image_id) <= tr_ids} "
              f"val_test_identical={set(sv.image_id) == set(vt.image_id)} "
              f"train classes={st.target_id.nunique()} persons={st.person_id.nunique()} clips={st.clip_id.nunique()} "
              f"whole_clips={whole_clips} imgs/clip={dict(Counter(st.groupby('clip_id').size()))}")
        print("      train per class:", st.groupby("target_id").size().to_dict())
    print("25⊂50:", set(subs[25].image_id) <= set(subs[50].image_id),
          "| 50⊂100:", set(subs[50].image_id) <= set(subs[100].image_id),
          "| 100 train part == train:", set(subs[100].loc[subs[100].split == "train", "image_id"]) == tr_ids)
    print("100pct rows identical to metadata.csv:", subs[100].equals(md) if len(subs[100]) == len(md) else "n/a (row count differs)")

    section("7. class_counts.csv 대조")
    cc = pd.read_csv(raw / "class_counts.csv", encoding="utf-8-sig")
    calc = md.groupby(["target_id", "split"]).agg(images=("image_id", "size"), people=("person_id", "nunique")).reset_index()
    m = cc.merge(calc, on=["target_id", "split"], suffixes=("_csv", "_calc"))
    print("rows compared:", len(m), "| images match:", bool((m.images_csv == m.images_calc).all()),
          "| people match:", bool((m.people_csv == m.people_calc).all()))
    cm = pd.read_csv(raw / "class_mapping.csv", encoding="utf-8-sig")
    pairs = set(map(tuple, md[["target_id", "exercise"]].drop_duplicates().values.tolist()))
    print("class_mapping consistent:", pairs == set(map(tuple, cm.values.tolist())))

    section("8. Train만: 클래스별 구성과 지름길 위험")
    tr = md[md.split == "train"]
    t = tr.groupby(["target_id", "exercise"]).agg(images=("image_id", "size"), clips=("clip_id", "nunique"),
                                                 persons=("person_id", "nunique"),
                                                 cameras=("camera", lambda x: "".join(sorted(set(x)))),
                                                 equipment=("equipment", lambda x: "/".join(sorted(set(x)))),
                                                 pose=("pose", lambda x: "/".join(sorted(set(x)))))
    print(t.to_string())
    print("\nTrain imbalance max/min images:", int(t.images.max()), "/", int(t.images.min()),
          "=", round(t.images.max() / t.images.min(), 2))
    print("Train camera counts:", dict(Counter(tr["camera"])))
    print("\nTrain person × class (images):")
    print(pd.crosstab(tr["person_id"], tr["target_id"]).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
