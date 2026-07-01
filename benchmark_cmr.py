#!/usr/bin/env python3
"""
CMR benchmark against /home/amar/Desktop/testset/
Filenames encode partial ground truth:
  Naklad <address>.jpg  → loading, address expected in box4_place_and_date_taking_over
  Razklad <address>.jpg → delivery, address expected in box3_place_of_delivery
  Others                → run OCR, report fill rate only

Metrics per image:
  fill_rate  — non-null fields / 24
  address_ok — whether the expected address substring appears in the relevant box
"""

import sys, os, re, time, json, difflib

sys.path.insert(0, "/media/amar/win/sledat-api")
import func

cfg = json.load(open("/media/amar/win/bic-datalab/config.json"))
func.DATALAB_API_KEY = cfg["datalab_key"]

TESTSET  = "/home/amar/Desktop/testset"
CMR_FIELDS = func.CMR_FIELDS

def parse_filename(fname):
    stem = os.path.splitext(fname)[0]
    stem_clean = stem.replace('_', ' ')
    if stem_clean.lower().startswith('naklad '):
        return 'naklad', stem_clean[7:].strip()
    if stem_clean.lower().startswith('razklad '):
        return 'razklad', stem_clean[8:].strip()
    return None, None

def addr_match(expected, actual):
    """True if expected address tokens are mostly present in actual text."""
    if not actual:
        return False
    e = expected.lower()
    a = actual.lower()
    # Simple: check if the street name (first word) and postcode/city appear
    tokens = re.split(r'[\s,_/-]+', e)
    tokens = [t for t in tokens if len(t) > 2]
    hits = sum(1 for t in tokens if t in a)
    return hits >= max(2, len(tokens) // 2)

def run_cmr_benchmark():
    images = sorted(f for f in os.listdir(TESTSET)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    total_fill   = []
    address_results = []

    print(f"Running {len(images)} CMR images through DataLabs balanced pipeline...\n")

    all_results = {}

    for fname in images:
        kind, expected_addr = parse_filename(fname)
        path = os.path.join(TESTSET, fname)
        image_bytes = open(path, 'rb').read()

        t0 = time.time()
        try:
            result  = func.run_pipeline(image_bytes, mode="cmr")
            elapsed = time.time() - t0
            fields  = result.get("fields", {})

            filled     = sum(1 for v in fields.values() if v)
            fill_rate  = filled / len(CMR_FIELDS) * 100
            total_fill.append(fill_rate)
            all_results[fname] = fields

            addr_ok = None
            if kind == 'naklad':
                addr_ok = addr_match(expected_addr, fields.get('box4_place_and_date_taking_over') or '')
                addr_label = f"box4 addr {'OK' if addr_ok else 'MISS'}"
            elif kind == 'razklad':
                addr_ok = addr_match(expected_addr, fields.get('box3_place_of_delivery') or '')
                addr_label = f"box3 addr {'OK' if addr_ok else 'MISS'}"
            else:
                addr_label = "no addr check"

            if kind:
                address_results.append(addr_ok)

            print(f"  {fname}")
            print(f"    fill={filled}/{len(CMR_FIELDS)} ({fill_rate:.0f}%)  {addr_label}  ({elapsed:.1f}s)")
            if kind == 'naklad':
                print(f"    expected box4: {expected_addr}")
                print(f"    got      box4: {fields.get('box4_place_and_date_taking_over') or '(empty)'}")
            elif kind == 'razklad':
                print(f"    expected box3: {expected_addr}")
                print(f"    got      box3: {fields.get('box3_place_of_delivery') or '(empty)'}")
            print()

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERR  {fname}: {e}  ({elapsed:.1f}s)\n")

    # Save full results for manual inspection
    out_path = "/home/amar/Desktop/cmr_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    avg_fill = sum(total_fill) / len(total_fill) if total_fill else 0
    addr_pass = sum(1 for x in address_results if x)
    addr_total = len(address_results)
    print(f"Average fill rate:    {avg_fill:.1f}%  ({len(CMR_FIELDS)} fields per doc)")
    print(f"Address check:        {addr_pass}/{addr_total} correct")
    print(f"Full results saved:   {out_path}")

if __name__ == "__main__":
    run_cmr_benchmark()
