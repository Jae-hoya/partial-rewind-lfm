"""SPEC §2.1 — HF wikimedia/structured-wikipedia 에서 영어판 파일럿 코퍼스를 만든다.

상류 데이터는 `enwiki/data/enwiki_namespace_0_*.parquet` 86 샤드, 37GB 다.
M1 은 `config.data.max_articles`(=30,000) 만 필요하다.

**샤드 하나만 받으면 안 된다.** 상류 행은 name 사전순으로 정렬돼 있어서 샤드 0 은
`"Good day, fellow!"` ~ `Custic` 구간뿐이다. 그 코퍼스로는 D 이후 문자로 시작하는
주제(예: quantum computing, vaccine)가 통째로 없어 교차 도메인 토픽 생성이 불가능하다.

따라서 **전 샤드에 걸친 층화 표집**을 한다:
  샤드마다 row group 을 시드 고정으로 골라 HTTP range read 로만 읽고(전체 다운로드 금지),
  샤드별 할당량만큼 시드 표집해 `data/corpus.parquet` 에 이어 쓴다.
전송량은 37GB 가 아니라 수백 MB 수준이고, 알파벳 전 구간이 균등하게 들어온다.

결정성: 같은 seed + 같은 revision -> 같은 (샤드, row group, 행) 선택. 재실행하면 같은 코퍼스.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402

SHARD_RE = re.compile(r"_(\d+)\.parquet$")


def list_shards(repo: str, revision: str, glob_prefix: str) -> list[str]:
    """상류 샤드 경로를 샤드 번호 오름차순으로. 목록 자체가 결정론적이어야 한다."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=revision, files_metadata=False)
    files = [s.rfilename for s in info.siblings if s.rfilename.startswith(glob_prefix)]
    files = [f for f in files if SHARD_RE.search(f)]
    if not files:
        raise SystemExit("상류 샤드를 찾지 못했다: " + repo + " / " + glob_prefix)
    return sorted(files, key=lambda f: int(SHARD_RE.search(f).group(1)))


def quotas(total: int, n_bins: int) -> list[int]:
    """total 을 n_bins 에 결정론적으로 배분. 앞쪽 bin 이 나머지를 하나씩 더 갖는다."""
    base, rem = divmod(total, n_bins)
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def shard_uri(repo: str, path: str, revision: str) -> str:
    return "datasets/" + repo + "@" + revision + "/" + path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="파일럿 코퍼스 다운로드 (SPEC §2.1)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="기본 data/corpus.parquet")
    ap.add_argument("--limit-shards", type=int, default=0, help="시험 실행용. 0 이면 전 샤드")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    repo = cfg["data.hf_dataset"]
    revision = cfg["data.hf_revision"]
    prefix = cfg["data.hf_file_glob"].split("*")[0]
    target = int(cfg["data.max_articles"])
    rg_per_shard = int(cfg.get("data.row_groups_per_shard", 2))
    min_abs = int(cfg.get("data.min_abstract_chars", 200))

    out = Path(args.out or (Path(cfg["paths.data_dir"]) / "corpus.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    shards = list_shards(repo, revision, prefix)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    per_shard = quotas(target, len(shards))
    ss = np.random.SeedSequence(seed).spawn(len(shards))

    print("repo=" + repo + " revision=" + revision[:12])
    print("샤드 " + str(len(shards)) + "개, 목표 " + str(target) + "건, "
          + "샤드당 row group " + str(rg_per_shard) + "개")

    writer: pq.ParquetWriter | None = None
    stats: list[dict[str, Any]] = []
    written = 0
    shortfall = 0
    t_start = time.time()

    try:
        for i, (path, quota, seq) in enumerate(zip(shards, per_shard, ss)):
            t0 = time.time()
            rng = np.random.default_rng(seq)
            with fs.open(shard_uri(repo, path, revision), "rb") as fh:
                pf = pq.ParquetFile(fh)
                n_rg = pf.metadata.num_row_groups
                k = min(rg_per_shard, n_rg)
                chosen = sorted(int(x) for x in rng.choice(n_rg, size=k, replace=False))
                tables = [pf.read_row_group(rg) for rg in chosen]
                tbl = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

            # 적격: abstract 가 충분히 길고 sections 가 비어 있지 않은 문서
            abstract = tbl.column("abstract").to_pylist()
            sections = tbl.column("sections").to_pylist()
            eligible = [
                j
                for j in range(tbl.num_rows)
                if abstract[j] and len(abstract[j]) >= min_abs and sections[j] and len(sections[j]) > 2
            ]
            take = min(quota, len(eligible))
            if take < quota:
                shortfall += quota - take
            picked = sorted(
                int(x) for x in rng.choice(np.array(eligible), size=take, replace=False)
            ) if take else []
            sub = tbl.take(pa.array(picked, type=pa.int64())) if picked else tbl.slice(0, 0)

            if writer is None:
                writer = pq.ParquetWriter(out, sub.schema, compression="zstd")
            writer.write_table(sub)
            written += sub.num_rows

            names = sub.column("name").to_pylist()
            stats.append(
                {
                    "shard": path,
                    "row_groups_total": n_rg,
                    "row_groups_read": chosen,
                    "rows_scanned": tbl.num_rows,
                    "rows_eligible": len(eligible),
                    "quota": quota,
                    "rows_taken": sub.num_rows,
                    "name_first": names[0] if names else None,
                    "name_last": names[-1] if names else None,
                    "seconds": round(time.time() - t0, 2),
                }
            )
            print(
                "[%3d/%3d] %s  rg=%s  %d/%d  %.1fs  %s"
                % (
                    i + 1,
                    len(shards),
                    path.rsplit("/", 1)[-1],
                    chosen,
                    sub.num_rows,
                    quota,
                    time.time() - t0,
                    (names[0][:28] if names else ""),
                ),
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t_start
    summary = {
        "repo": repo,
        "revision": revision,
        "config_name": cfg["data.hf_config"],
        "language": cfg["data.language"],
        "seed": seed,
        "target_articles": target,
        "written_articles": written,
        "shortfall": shortfall,
        "n_shards": len(shards),
        "row_groups_per_shard": rg_per_shard,
        "min_abstract_chars": min_abs,
        "elapsed_s": round(elapsed, 1),
        "output": str(out),
        "shards": stats,
    }
    stats_path = out.parent / "download_stats.json"
    with io.open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print()
    print("문서 " + str(written) + "건 -> " + str(out))
    print("소요 " + str(round(elapsed, 1)) + "s, 부족분 " + str(shortfall))
    print("통계 -> " + str(stats_path))
    if shortfall:
        print("경고: 할당량 미달. row_groups_per_shard 를 올려야 한다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
