import os
import sys
import argparse
import shutil
import time
import traceback
import numpy as np
import pandas as pd
import torch

DATASETS = [
    ("./data/ETT-small/", "ETTh1.csv", "csv"),
    ("./data/ETT-small/", "ETTh2.csv", "csv"),
    ("./data/ETT-small/", "ETTm1.csv", "csv"),
    ("./data/ETT-small/", "ETTm2.csv", "csv"),
    ("./data/weather/", "weather.csv", "csv"),
    ("./data/electricity/", "electricity.csv", "csv"),
    ("./data/exchange_rate/", "exchange_rate.csv", "csv"),
    ("./data/traffic/", "traffic.csv", "csv"),
    ("./data/illness/", "national_illness.csv", "csv"),
    ("./data/Solar/", "solar_AL.txt", "solar"),
    ("./data/PEMS/", "PEMS03.npz", "pems"),
    ("./data/PEMS/", "PEMS04.npz", "pems"),
    ("./data/PEMS/", "PEMS07.npz", "pems"),
    ("./data/PEMS/", "PEMS08.npz", "pems"),
]


def convert_csv(root_path, data_path):
    fpath = os.path.join(root_path, data_path)
    print(f"  Reading CSV: {fpath}")
    t0 = time.time()
    df_raw = pd.read_csv(fpath)
    all_values = df_raw[df_raw.columns[1:]].values.astype(np.float64)
    all_dates = pd.to_datetime(df_raw['date']).values
    columns = list(df_raw.columns)
    elapsed = time.time() - t0
    print(f"    Shape: {all_values.shape}, dates: {all_dates.shape}, "
          f"columns: {columns[:3]}...({len(columns)}), {elapsed:.2f}s")
    return {
        'values': all_values,
        'dates': all_dates,
        'columns': columns,
    }


def convert_solar(root_path, data_path):
    fpath = os.path.join(root_path, data_path)
    print(f"  Reading Solar TXT: {fpath}")
    t0 = time.time()
    rows = []
    with open(fpath, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split(',')
            rows.append(np.array([float(x) for x in parts]))
    all_values = np.stack(rows, 0).astype(np.float64)
    elapsed = time.time() - t0
    print(f"    Shape: {all_values.shape}, {elapsed:.2f}s")
    return {'values': all_values}


def convert_pems(root_path, data_path):
    fpath = os.path.join(root_path, data_path)
    print(f"  Reading PEMS NPZ: {fpath}")
    t0 = time.time()
    raw = np.load(fpath, allow_pickle=True)
    all_values = raw['data'][:, :, 0].astype(np.float64)
    elapsed = time.time() - t0
    print(f"    Shape: {all_values.shape}, {elapsed:.2f}s")
    return {'values': all_values}


CONVERTERS = {
    'csv': convert_csv,
    'solar': convert_solar,
    'pems': convert_pems,
}


def pt_path_for(root_path, data_path):
    base, _ = os.path.splitext(data_path)
    return os.path.join(root_path, base + '.pt')


def convert_all(force=False):
    print("=" * 60)
    print("Converting datasets to .pt format")
    print("=" * 60)
    success, skipped, failed = 0, 0, 0
    for root_path, data_path, dtype in DATASETS:
        out_path = pt_path_for(root_path, data_path)
        src_path = os.path.join(root_path, data_path)
        if not os.path.exists(src_path):
            print(f"  SKIP (source not found): {src_path}")
            skipped += 1
            continue
        if os.path.exists(out_path) and not force:
            print(f"  SKIP (exists): {out_path}")
            skipped += 1
            continue
        try:
            converter = CONVERTERS[dtype]
            data = converter(root_path, data_path)
            t0 = time.time()
            torch.save(data, out_path)
            save_time = time.time() - t0
            fsize_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"    Saved: {out_path} ({fsize_mb:.1f} MB, {save_time:.2f}s)")
            success += 1
        except Exception as e:
            print(f"  FAIL: {src_path} -> {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"Done: {success} converted, {skipped} skipped, {failed} failed")
    print("=" * 60)
    return failed == 0


def _clear_pycache():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "src", "data_provider", "__pycache__")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  Cleared stale cache: {cache_dir}")


def verify_all():
    print("=" * 60)
    print("Verifying .pt equivalence")
    print("=" * 60)

    _clear_pycache()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.data_provider.data_loader import (
        Dataset_ETT_hour,
        Dataset_ETT_minute,
        Dataset_Custom,
        Dataset_PEMS,
        Dataset_Solar,
    )

    all_pass = True

    verify_cases = [
        ("ETTh1", Dataset_ETT_hour, "./data/ETT-small/", "ETTh1.csv",
         {'features': 'M', 'target': 'OT'}),
        ("ETTh2", Dataset_ETT_hour, "./data/ETT-small/", "ETTh2.csv",
         {'features': 'M', 'target': 'OT'}),
        ("ETTm1", Dataset_ETT_minute, "./data/ETT-small/", "ETTm1.csv",
         {'features': 'M', 'target': 'OT'}),
        ("ETTm2", Dataset_ETT_minute, "./data/ETT-small/", "ETTm2.csv",
         {'features': 'M', 'target': 'OT'}),
        ("Weather", Dataset_Custom, "./data/weather/", "weather.csv",
         {'features': 'M', 'target': 'OT'}),
        ("ECL", Dataset_Custom, "./data/electricity/", "electricity.csv",
         {'features': 'M', 'target': 'OT'}),
        ("Exchange", Dataset_Custom, "./data/exchange_rate/", "exchange_rate.csv",
         {'features': 'M', 'target': 'OT'}),
        ("Traffic", Dataset_Custom, "./data/traffic/", "traffic.csv",
         {'features': 'M', 'target': 'OT'}),
        ("Solar", Dataset_Solar, "./data/Solar/", "solar_AL.txt",
         {}),
        ("PEMS03", Dataset_PEMS, "./data/PEMS/", "PEMS03.npz",
         {}),
        ("PEMS04", Dataset_PEMS, "./data/PEMS/", "PEMS04.npz",
         {}),
        ("PEMS07", Dataset_PEMS, "./data/PEMS/", "PEMS07.npz",
         {}),
        ("PEMS08", Dataset_PEMS, "./data/PEMS/", "PEMS08.npz",
          {}),
        ("Illness", Dataset_Custom, "./data/illness/", "national_illness.csv",
          {'features': 'M', 'target': 'OT'}),
    ]

    for name, cls, root, dpath, extra in verify_cases:
        pt_file = pt_path_for(root, dpath)
        if not os.path.exists(pt_file):
            print(f"\n  SKIP (no .pt): {pt_file}")
            continue

        src_file = os.path.join(root, dpath)
        if not os.path.exists(src_file):
            print(f"\n  SKIP (no source): {src_file}")
            continue

        print(f"\n  Verifying: {name}")

        base_kwargs = dict(
            flag='train',
            size=[96, 48, 96],
            timeenc=0,
            freq='h',
        )
        base_kwargs.update(extra)

        bak_file = pt_file + ".bak"
        try:
            os.rename(pt_file, bak_file)

            ds_csv = cls(root_path=root, data_path=dpath, **base_kwargs)

            os.rename(bak_file, pt_file)
            bak_file = None

            ds_pt = cls(root_path=root, data_path=dpath, **base_kwargs)

            ok = True
            for attr in ['data_x', 'data_y', 'data_stamp']:
                if not hasattr(ds_csv, attr) or not hasattr(ds_pt, attr):
                    continue
                a = getattr(ds_csv, attr)
                b = getattr(ds_pt, attr)
                if a.shape != b.shape:
                    print(f"    FAIL {attr} shape: {a.shape} vs {b.shape}")
                    ok = False
                    continue
                if not torch.allclose(a, b, atol=1e-6):
                    diff = (a - b).abs().max().item()
                    print(f"    FAIL {attr} max_diff={diff:.2e}")
                    ok = False

            if not np.allclose(ds_csv.scaler.mean_, ds_pt.scaler.mean_, atol=1e-10):
                diff = np.abs(ds_csv.scaler.mean_ - ds_pt.scaler.mean_).max()
                print(f"    FAIL scaler.mean_ max_diff={diff:.2e}")
                ok = False
            if not np.allclose(ds_csv.scaler.scale_, ds_pt.scaler.scale_, atol=1e-10):
                diff = np.abs(ds_csv.scaler.scale_ - ds_pt.scaler.scale_).max()
                print(f"    FAIL scaler.scale_ max_diff={diff:.2e}")
                ok = False
            if len(ds_csv) != len(ds_pt):
                print(f"    FAIL __len__: {len(ds_csv)} vs {len(ds_pt)}")
                ok = False

            if ok:
                print(f"    PASS")
            else:
                all_pass = False
        except Exception:
            print(f"    ERROR:")
            traceback.print_exc()
            all_pass = False
        finally:
            if bak_file and os.path.exists(bak_file):
                if not os.path.exists(pt_file):
                    os.rename(bak_file, pt_file)
                else:
                    os.remove(bak_file)

    print("\n" + "=" * 60)
    if all_pass:
        print("All verifications PASSED")
    else:
        print("Some verifications FAILED")
    print("=" * 60)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to .pt format")
    parser.add_argument("--verify", action="store_true",
                        help="Verify equivalence after conversion")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification (no conversion)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .pt files")
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_all()
        sys.exit(0 if ok else 1)

    ok = convert_all(force=args.force)

    if args.verify and ok:
        ok = verify_all()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
