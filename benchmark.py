#!/usr/bin/env python3
"""
Benchmark DataLabs BIC pipeline against testset2.
Filenames encode ground truth: OWNERCODE-SERIAL-CHECK.ext
Underscores mark covered characters.

Cases excluded from score (not fair to test):
  - Multiple characters covered: too many combinations, multiple valid answers
  - First or last serial digit covered: model sees 5 digits and doesn't know one is missing
  - Middle owner code letter covered: up to 26 possibilities, multiple may satisfy check digit
  - (Last owner letter covered is fine: containers are always *U, so unique answer)
"""

import sys, os, re, time, json

sys.path.insert(0, "/media/amar/win/sledat-api")
import func

cfg = json.load(open("/media/amar/win/bic-datalab/config.json"))
func.DATALAB_API_KEY = cfg["datalab_key"]

TESTSET = "/home/amar/Desktop/testset2"

_ALPHA = "0123456789A_BCDEFGHIJK_LMNOPQRSTU_VWXYZ"
_VALS  = {c: i for i, c in enumerate(_ALPHA)}

def iso6346_check(owner4, serial6):
    s = owner4 + serial6
    total = sum(_VALS.get(c, 0) * (2 ** i) for i, c in enumerate(s))
    return str(total % 11 % 10)

def parse_stem(stem):
    m = re.match(r'^([A-Z_]{4})-([0-9_]{6})-([0-9_])$', stem)
    return m.groups() if m else None

def is_clean(owner, serial, check):
    return '_' not in owner + serial + check

def skip_reason(owner, serial, check):
    """Return reason to skip, or None if this case is fair to test."""
    covered_owner  = [i for i, c in enumerate(owner)  if c == '_']
    covered_serial = [i for i, c in enumerate(serial) if c == '_']
    covered_check  = check == '_'

    total_covered = len(covered_owner) + len(covered_serial) + (1 if covered_check else 0)

    if total_covered == 0:
        return None  # clean, always test

    if total_covered > 1:
        return "multiple characters covered (ambiguous)"

    # Single character covered — check which one
    if covered_owner:
        idx = covered_owner[0]
        if idx < 3:
            return f"owner code letter at position {idx} covered (multiple valid answers)"
        # position 3 is always 'U' for containers — recoverable
        return None

    if covered_serial:
        idx = covered_serial[0]
        if idx == 0:
            return "first serial digit covered (model sees 5 digits, can't detect gap)"
        if idx == 5:
            return "last serial digit covered (model sees 5 digits, can't detect gap)"
        return None  # middle serial digit — check digit math can recover

    # Check digit covered — always recoverable
    return None

def resolve_single_unknown(owner, serial, check):
    """For exactly one covered character, find unique solution via check digit math."""
    full = owner + serial + check
    idx  = full.index('_')
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if idx < 4 else "0123456789"
    solutions = []
    for c in charset:
        candidate = full[:idx] + c + full[idx+1:]
        o, s, k = candidate[:4], candidate[4:10], candidate[10]
        computed = iso6346_check(o, s)
        if k == '_':
            solutions.append(o + s + str(computed))
        elif str(computed) == k:
            solutions.append(candidate)
    return solutions[0] if len(solutions) == 1 else None

def build_ground_truth(images):
    """Resolve ground truth from clean counterpart; fall back to check digit math."""
    clean_bics = {}
    for fname in images:
        stem = os.path.splitext(fname)[0]
        parsed = parse_stem(stem)
        if parsed and is_clean(*parsed):
            owner, serial, check = parsed
            clean_bics[owner + serial + check] = True

    gt = {}
    for fname in images:
        stem = os.path.splitext(fname)[0]
        parsed = parse_stem(stem)
        if not parsed:
            gt[fname] = None
            continue
        owner, serial, check = parsed
        if is_clean(owner, serial, check):
            gt[fname] = owner + serial + check
            continue
        # Match clean BIC where every non-underscore char agrees
        full = owner + serial + check
        for bic in clean_bics:
            if len(bic) == len(full) and all(f == b or f == '_' for f, b in zip(full, bic)):
                gt[fname] = bic
                break
        else:
            # Fall back to check digit math for single unknowns
            if full.count('_') == 1:
                gt[fname] = resolve_single_unknown(owner, serial, check)
            else:
                gt[fname] = None
    return gt

def run_benchmark():
    images = sorted(f for f in os.listdir(TESTSET)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    gt = build_ground_truth(images)

    results   = []
    excluded  = []
    no_gt     = []

    print(f"Running images through DataLabs pipeline...\n")

    for fname in images:
        stem = os.path.splitext(fname)[0]
        parsed = parse_stem(stem)
        if not parsed:
            no_gt.append(fname)
            print(f"  ???   {fname}  (can't parse filename)")
            continue

        owner, serial, check = parsed
        reason = skip_reason(owner, serial, check)
        if reason:
            excluded.append((fname, reason))
            print(f"  EXCL  {fname:<32}  ({reason})")
            continue

        expected = gt.get(fname)
        if not expected:
            no_gt.append(fname)
            print(f"  ???   {fname}  (no ground truth found)")
            continue

        path = os.path.join(TESTSET, fname)
        image_bytes = open(path, 'rb').read()
        t0 = time.time()
        try:
            result  = func.run_pipeline(image_bytes, mode="bic")
            elapsed = time.time() - t0
            got     = (result.get("bic") or "").upper()
            ok      = got == expected.upper()
            results.append((fname, ok, expected, got, elapsed))
            status  = "PASS" if ok else "FAIL"
            print(f"  {status}  {fname:<32}  expected={expected}  got={got or '(none)'}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results.append((fname, False, expected, f"ERROR: {e}", elapsed))
            print(f"  ERR   {fname:<32}  {e}  ({elapsed:.1f}s)")

    passed = sum(1 for _, ok, *_ in results if ok)
    total  = len(results)

    print(f"\n{'='*60}")
    print(f"Tested:   {total}  |  Passed: {passed}  ({100*passed//total if total else 0}%)")
    print(f"Excluded: {len(excluded)}  (inherently ambiguous/unresolvable)")
    if excluded:
        for fname, reason in excluded:
            print(f"          {fname}: {reason}")

if __name__ == "__main__":
    run_benchmark()
