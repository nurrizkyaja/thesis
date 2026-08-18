import hashlib
from pathlib import Path
from collections import defaultdict
import csv
import json
from datetime import datetime

def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Menghitung hash SHA-256 dari sebuah file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_dataset_duplicates(
    data_split_dir: Path, 
    output_txt: Path = None, 
    output_csv: Path = None,
    output_json: Path = None
):
    """
    Mengecek duplikasi file di dalam folder data/split dan menyimpan hasilnya ke file.
    
    Parameters:
        data_split_dir (Path): Path ke direktori data/split.
        output_txt (Path): Path file teks (.txt) untuk ringkasan laporan.
        output_csv (Path): Path file CSV (.csv) untuk detail duplikat (mudah dibuka di Excel).
        output_json (Path): Path file JSON (.json) untuk otomatisasi pembersihan data.
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    hash_map = defaultdict(list)
    
    print(f"Memindai direktori: {data_split_dir.resolve()}...\n")
    
    all_images = [p for p in data_split_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_extensions]
    
    for img_path in all_images:
        h = compute_sha256(img_path)
        rel_path = img_path.relative_to(data_split_dir)
        hash_map[h].append(rel_path)
        
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    
    cross_subset_leaks = 0
    lines = []
    
    lines.append("=" * 60)
    lines.append("HASIL ANALISIS DUPLIKASI SHA-256")
    lines.append("=" * 60)
    lines.append(f"Waktu Analisis       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Direktori Dataset    : {data_split_dir.resolve()}")
    lines.append(f"Total Citra Dipindai : {len(all_images)}")
    lines.append(f"Total Hash Unik      : {len(hash_map)}")
    lines.append(f"Total File Duplikat  : {sum(len(v) for v in duplicates.values()) - len(duplicates)}")
    lines.append(f"Total Kelompok Hash  : {len(duplicates)}\n")
    
    csv_rows = []
    
    if duplicates:
        lines.append("--- Rincian Duplikasi Lintas Subset (Train/Val/Test) ---")
        for h, paths in duplicates.items():
            subsets = set(p.parts[0] for p in paths)
            is_leak = len(subsets) > 1
            if is_leak:
                cross_subset_leaks += 1
                lines.append(f"\n[LEAK DETECTED] Hash: {h[:12]}...")
                for p in paths:
                    lines.append(f"  - {p}")
            
            for p in paths:
                csv_rows.append({
                    "sha256_hash": h,
                    "subset": p.parts[0] if len(p.parts) > 1 else "",
                    "relative_path": str(p),
                    "is_cross_subset_leak": is_leak
                })
        
        if cross_subset_leaks == 0:
            lines.append("\n>> Aman! Seluruh duplikasi hanya terjadi di dalam subset yang sama (tidak ada data leakage lintas Train/Val/Test).")
        else:
            lines.append(f"\n>> Ditemukan {cross_subset_leaks} kelompok hash yang bocor lintas subset!")
    else:
        lines.append(">> Bersih! Tidak ditemukan file duplikat biner sama sekali di dalam dataset.")

    report_text = "\n".join(lines)
    
    print(report_text)
    
    if output_txt:
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\n[INFO] Laporan ringkasan disimpan ke: {output_txt.resolve()}")

    if output_csv and csv_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha256_hash", "subset", "relative_path", "is_cross_subset_leak"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[INFO] Detail duplikat disimpan ke CSV: {output_csv.resolve()}")

    if output_json and duplicates:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        json_data = {
            "summary": {
                "total_scanned": len(all_images),
                "unique_hashes": len(hash_map),
                "duplicate_files_count": sum(len(v) for v in duplicates.values()) - len(duplicates),
                "duplicate_groups_count": len(duplicates),
                "cross_subset_leaks_count": cross_subset_leaks
            },
            "duplicates": {
                h: [str(p) for p in paths] for h, paths in duplicates.items()
            }
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
        print(f"[INFO] Detail duplikat disimpan ke JSON: {output_json.resolve()}")

if __name__ == "__main__":
    PROJECT_ROOT = Path.cwd()
    DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "split"
    OUTPUT_TXT = PROJECT_ROOT / "dataset_duplicate_report.txt"
    OUTPUT_CSV = PROJECT_ROOT / "dataset_duplicates.csv"
    OUTPUT_JSON = PROJECT_ROOT / "dataset_duplicates.json"
    
    if DATA_SPLIT_DIR.exists():
        check_dataset_duplicates(
            DATA_SPLIT_DIR,
            output_txt=OUTPUT_TXT,
            output_csv=OUTPUT_CSV,
            output_json=OUTPUT_JSON
        )
    else:
        print(f"Folder tidak ditemukan: {DATA_SPLIT_DIR}")