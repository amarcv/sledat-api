# BIC and CMR extraction via Datalab.to API.
# Requires DATALAB_API_KEY (set in config.json or as env var).
#
# Extraction modes (datalab.to/benchmark/overall):
#   fast     ~$6/1K pages,  ~10-30s  - used for BIC
#   balanced ~$25/1K pages, ~30-90s  - used for CMR

import io
import json
import logging
import os
import re
import time
from itertools import product as iproduct

import cv2
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def crop_document(image_bytes: bytes) -> bytes:
    """Find the document outline and warp it flat. Returns original if no clear outline found."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    h, w = img.shape[:2]
    min_area = h * w * 0.05  # document must cover at least 5% of frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    doc_contour = None
    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        # try progressively looser approximation
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                doc_contour = approx
                break
        if doc_contour is not None:
            break
        # if still not 4 corners, try convex hull of the contour
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                doc_contour = approx
                break
        if doc_contour is not None:
            break

    if doc_contour is None:
        log.info("crop_document: no document contour found, returning original")
        return image_bytes

    pts = doc_contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))

    _, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"crop_document: warped {w}x{h} → {max_w}x{max_h}")
    return buf.tobytes()


DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY", "")
_BASE          = "https://www.datalab.to/api/v1"
_POLL_INTERVAL = 0.5  # seconds between status polls
_POLL_TIMEOUT  = 120  # seconds before TimeoutError

# When True, _spaced_wildcards() also handles two covered serial digits.
RECOVER_TWO_MISSING: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ISO 6346 check digit math
# ─────────────────────────────────────────────────────────────────────────────

# Letters A-Z map to 10..38 skipping every multiple of 11 (11, 22, 33 are absent).
_CVAL: dict[str, int] = {str(d): d for d in range(10)}
_v = 10
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _CVAL[_ch] = _v
    _v += 1
    if _v % 11 == 0:
        _v += 1


def _compute_check(prefix10: str) -> int:
    """Compute the ISO 6346 check digit for a 10-character owner+serial string."""
    total = sum(_CVAL[c] * (2 ** i) for i, c in enumerate(prefix10))
    return (total % 11) % 10


def is_valid_bic(code: str) -> bool:
    """Return True if code is a syntactically valid and check-digit-correct BIC."""
    if not re.fullmatch(r"[A-Z]{4}\d{7}", code):
        return False
    return _compute_check(code[:10]) == int(code[10])


# ─────────────────────────────────────────────────────────────────────────────
# BIC recovery helpers (pure math — no model dependency)
# ─────────────────────────────────────────────────────────────────────────────

_BIC_EXACT = re.compile(r"[A-Z]{4}\d{7}")
_BIC_LOOSE = re.compile(r"[A-Z]{3,4}[\s\-]*[A-Z]?[\s\-]*[\d][\d\s]{5,8}")


def _condense(text: str) -> str:
    return re.sub(r"[\s\-_\.]", "", text.upper())


def _interpret_dashed_parts(parts: list[str]) -> list[str]:
    """Convert dash-split BIC parts into wildcard candidate strings.

    Extra '-' within owner or serial marks a missing character; each gap → '?'.
    """
    if len(parts) < 3:
        return None
    if not re.fullmatch(r'\d', parts[-1]):
        return None
    check = parts[-1]
    owner_parts, serial_parts = [], []
    i = 0
    while i < len(parts) - 1 and re.fullmatch(r'[A-Z?]+', parts[i]):
        owner_parts.append(parts[i])
        i += 1
    while i < len(parts) - 1 and re.fullmatch(r'[\d?]+', parts[i]):
        serial_parts.append(parts[i])
        i += 1
    if i != len(parts) - 1 or not owner_parts or not serial_parts:
        return None
    owner_total  = sum(len(p) for p in owner_parts)
    serial_total = sum(len(p) for p in serial_parts)
    if owner_total > 4 or serial_total > 6:
        return []
    if len(owner_parts) == 1:
        if owner_total == 4:
            owner_strs = [owner_parts[0]]
        elif owner_total == 3:
            # Missing letter could be at any of the 4 positions
            s = owner_parts[0]
            owner_strs = ['?' + s, s[0] + '?' + s[1:], s[:2] + '?' + s[2], s + '?']
        else:
            return []
    else:
        owner_str = '?'.join(owner_parts)
        if len(owner_str) != 4:
            return []
        owner_strs = [owner_str]
    if len(serial_parts) == 1:
        if serial_total != 6:
            return []
        serial_str = serial_parts[0]
    else:
        serial_str = '?'.join(serial_parts)
        if len(serial_str) != 6:
            return []
    results = []
    for owner_str in owner_strs:
        candidate = owner_str + serial_str + check
        if len(candidate) == 11:
            results.append(candidate)
    return results


def _recover_dashed_serial_gap(owner: str, partial_serial: str, check_str: str) -> list[str]:
    """Try inserting one digit at the front or back of a 5-digit serial."""
    results, seen = [], set()
    for pos in (0, len(partial_serial)):
        for d in "0123456789":
            full_serial = partial_serial[:pos] + d + partial_serial[pos:]
            if len(full_serial) == 6:
                bic = owner + full_serial + check_str
                if bic not in seen and is_valid_bic(bic):
                    seen.add(bic)
                    results.append(bic)
    return results


def _parse_dashed_bic(text: str) -> tuple[list[str], list[str], set[str]]:
    """Parse dashed BIC patterns (XXXX-YYYYYY-Z) from OCR text.

    Returns (wildcard_candidates, serial_gap_bics, blocked_condensed_forms).
    """
    upper = text.upper()
    wildcard_cands, serial_gap_bics, blocked_forms = [], [], set()
    seen_cands: set[str] = set()

    for m in re.finditer(r'[A-Z]{4}-\d{5,6}-[\d?]', upper):
        blocked_forms.add(re.sub(r'[^A-Z0-9]', '', m.group()))

    for m in re.finditer(r'[A-Z][A-Z0-9?-]{7,20}[\d?]', upper):
        token = m.group()
        if token.count('-') < 2:
            continue
        parts = [p for p in token.split('-') if p]
        if len(parts) < 3:
            continue

        # XXXX-YYYYYY-? : check digit is the last "serial" digit
        if (len(parts) == 3 and re.fullmatch(r'[A-Z]{4}', parts[0])
                and re.fullmatch(r'\d{6}', parts[1]) and parts[2] == '?'):
            owner, true_check = parts[0], parts[1][-1]
            for h in _recover_dashed_serial_gap(owner, parts[1][:5], true_check):
                if h not in seen_cands:
                    seen_cands.add(h); serial_gap_bics.append(h)
            continue

        # XXXX-YYYYY-Z : 5 serial digits + known check digit
        if (len(parts) == 3 and re.fullmatch(r'[A-Z]{4}', parts[0])
                and re.fullmatch(r'\d{5}', parts[1]) and re.fullmatch(r'\d', parts[2])):
            for h in _recover_dashed_serial_gap(parts[0], parts[1], parts[2]):
                if h not in seen_cands:
                    seen_cands.add(h); serial_gap_bics.append(h)
            continue

        # XXXX-YYYYYY-Z where check fails: last serial digit may be the real check
        if (len(parts) == 3 and re.fullmatch(r'[A-Z]{4}', parts[0])
                and re.fullmatch(r'\d{6}', parts[1]) and re.fullmatch(r'\d', parts[2])
                and not is_valid_bic(parts[0] + parts[1] + parts[2])):
            for h in _recover_dashed_serial_gap(parts[0], parts[1][:5], parts[2]):
                if h not in seen_cands:
                    seen_cands.add(h); serial_gap_bics.append(h)
            continue

        for cand in (_interpret_dashed_parts(parts) or []):
            if cand not in seen_cands:
                seen_cands.add(cand); wildcard_cands.append(cand)

    return wildcard_cands, serial_gap_bics, blocked_forms


def _spaced_wildcards(text: str) -> list[str]:
    """Infer ? positions from digit-group spacing gaps in OCR text.

    e.g. 'HPCU 400 35 9' → serial digits total 6, one gap → 'HPCU400?359'.
    """
    results = []
    upper = text.upper()
    valid_totals = (5, 6, 7) if RECOVER_TWO_MISSING else (6, 7)

    def _process(owner: str, tokens: list[str]) -> None:
        total = sum(len(t) for t in tokens)
        if total not in valid_totals:
            return
        if total == 7:
            s = ''.join(tokens)
            candidate = owner + s
            if len(candidate) == 11 and '?' in candidate:
                results.append(candidate)
            return
        digits = list(''.join(tokens))
        if total == 6:
            if len(tokens) < 2:
                # No visible gap — the 10-char prefix path handles this correctly.
                # Generating speculative edge-position wildcards here would block
                # that path via the no_wc seen-set and return a wrong BIC.
                return
            boundaries: list[int] = [0]
            cumulative = 0
            for t in tokens[:-1]:
                cumulative += len(t)
                boundaries.append(cumulative)
            if len(boundaries) > 2:
                # Multiple gaps: use only the internal boundary positions
                boundaries = boundaries[1:-1]
            elif len(boundaries) == 2:
                # Exactly one gap: insert '?' at that exact position
                boundaries = [boundaries[1]]
            else:
                boundaries = [0, 5]
            for b in boundaries:
                candidate = owner + ''.join(digits[:b] + ['?'] + digits[b:])
                if len(candidate) == 11:
                    results.append(candidate)
        elif total == 5:
            all_pos = list(range(len(digits) + 1))
            for i, b1 in enumerate(all_pos):
                for b2 in all_pos[i:]:
                    if b1 == b2:
                        chars = digits[:b1] + ['?', '?'] + digits[b1:]
                    else:
                        chars = digits[:b1] + ['?'] + digits[b1:b2] + ['?'] + digits[b2:]
                    candidate = owner + ''.join(chars)
                    if len(candidate) == 11:
                        results.append(candidate)

    for m in re.finditer(r'\b([A-Z]{4})\s+((?:[\d?]+\s*){2,8})', upper):
        owner = m.group(1)
        tokens = [t for t in re.split(r'\s+', m.group(2).strip()) if re.fullmatch(r'[\d?]+', t)]
        total = sum(len(t) for t in tokens)
        if total in valid_totals:
            _process(owner, tokens)
        elif total > max(valid_totals):
            for drop_i in range(len(tokens)):
                _process(owner, tokens[:drop_i] + tokens[drop_i + 1:])
    return results


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _recover_wildcards_all(candidate: str) -> list[str]:
    """Return all valid BICs for '?' or wrong-type positions in the candidate."""
    chars = list(candidate)
    wild = [i for i, c in enumerate(chars) if c == "?" or (i >= 4 and not c.isdigit())]
    if not wild:
        return []
    charset = lambda pos: "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if pos < 4 else "0123456789"
    results = []
    for vals in iproduct(*(charset(p) for p in wild)):
        test = chars.copy()
        for p, v in zip(wild, vals):
            test[p] = v
        bic = "".join(test)
        if is_valid_bic(bic):
            results.append(bic)
    return results


def _insert_candidates(short: str) -> list[str]:
    """Return all valid BICs formed by inserting one character into a 10-char string."""
    if len(short) != 10 or not re.fullmatch(r"[A-Z]{3}", short[:3]):
        return []
    results, seen = [], set()
    for pos in range(11):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if pos < 4 else "0123456789"
        for ch in chars:
            candidate = short[:pos] + ch + short[pos:]
            if candidate not in seen and is_valid_bic(candidate):
                seen.add(candidate); results.append(candidate)
    return results


def _brute_force_owner(candidate: str) -> tuple[list[str], int] | None:
    """Search all 26^4 owner codes with the candidate's serial+check held fixed."""
    if len(candidate) != 11 or not re.fullmatch(r"[A-Z]{4}\d{7}", candidate):
        return None
    owner_ocr = candidate[:4]
    serial    = candidate[4:10]
    check_c   = candidate[10]
    ocr_check = int(check_c)
    serial_sum  = sum(_CVAL[c] * (2 ** (i + 4)) for i, c in enumerate(serial))
    check_raws  = [ocr_check] if ocr_check > 0 else [0, 10]
    needed_mods = {(r - serial_sum % 11) % 11 for r in check_raws}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    best_dist = 99
    buckets: dict[int, list[str]] = {}
    for l0 in letters:
        v0 = _CVAL[l0]
        for l1 in letters:
            v1 = _CVAL[l1] * 2
            for l2 in letters:
                v2 = _CVAL[l2] * 4
                for l3 in letters:
                    if (v0 + v1 + v2 + _CVAL[l3] * 8) % 11 not in needed_mods:
                        continue
                    new_owner = l0 + l1 + l2 + l3
                    bic = new_owner + serial + check_c
                    if not is_valid_bic(bic) or new_owner == owner_ocr:
                        continue
                    dist = sum(a != b for a, b in zip(owner_ocr, new_owner))
                    if dist <= best_dist:
                        if dist < best_dist:
                            best_dist = dist; buckets = {}
                        buckets.setdefault(dist, []).append(bic)
    if buckets:
        return buckets[best_dist], best_dist
    return None


def _brute_force(candidate: str) -> tuple[list[str], int] | None:
    """Return (candidates, edit_distance): valid BICs at minimum Hamming distance."""
    if len(candidate) != 11 or not re.fullmatch(r"[A-Z]{4}", candidate[:4]):
        return None
    owner      = candidate[:4]
    serial_ocr = candidate[4:10]
    ocr_check  = int(candidate[10]) if candidate[10].isdigit() else -1
    owner_sum  = sum(_CVAL[c] * (2 ** i) for i, c in enumerate(owner))

    if ocr_check >= 0:
        INV_LAST    = 2
        powers5     = [2 ** (i + 4) for i in range(5)]
        raw_targets = [ocr_check] if ocr_check > 0 else [0, 10]
        best_dist   = 99
        results: dict[int, list[str]] = {}
        for n5 in range(100_000):
            s5      = f"{n5:05d}"
            partial = (owner_sum + sum(_CVAL[c] * powers5[i] for i, c in enumerate(s5))) % 11
            for tgt in raw_targets:
                d6 = (INV_LAST * (tgt - partial)) % 11
                if d6 > 9:
                    continue
                serial = s5 + str(d6)
                bic    = owner + serial + str(ocr_check)
                if bic == candidate:
                    continue
                d = _hamming(serial_ocr, serial)
                if d <= best_dist:
                    if d < best_dist:
                        best_dist = d; results = {}
                    results.setdefault(d, []).append(bic)
        if results:
            return results[best_dist], best_dist

    powers6  = [2 ** (i + 4) for i in range(6)]
    best_key = (2, 99)
    results2: dict[tuple, list[str]] = {}
    for n in range(1_000_000):
        serial = f"{n:06d}"
        s_sum  = sum(_CVAL[c] * powers6[i] for i, c in enumerate(serial))
        check  = ((owner_sum + s_sum) % 11) % 10
        bic    = owner + serial + str(check)
        if bic == candidate:
            continue
        key = (0 if check == ocr_check else 1, _hamming(serial_ocr, serial))
        if key <= best_key:
            if key < best_key:
                best_key = key; results2 = {}
            results2.setdefault(key, []).append(bic)
    if not results2:
        return None
    return results2[best_key], best_key[1]


def extract_bic(ocr_text: str) -> tuple[str | list | None, str, int]:
    """Parse OCR text and return (bic, confidence, edit_distance).

    confidence values:
      'exact'        — OCR gave a valid BIC directly                  (distance 0)
      'recovered'    — one gap or error filled in, check confirmed     (distance 1)
      'brute_forced' — nearest-neighbour search, list of candidates    (distance ≥1)
      'not_found'
    """
    upper      = ocr_text.upper()
    condensed  = _condense(upper)
    alnum_only = re.sub(r"[^A-Z0-9]", "", condensed)
    wild_condensed = re.sub(r"[^A-Z0-9?]", "?", re.sub(r"[\s\-_\.]", "", upper))

    # Exact: valid BIC present in raw output.
    # ISO 6346: the 4th character is the equipment category identifier.
    # Freight containers always use 'U'. If OCR gives a valid BIC with a
    # non-U 4th letter we continue to recovery — it's almost certainly a
    # misread (e.g. 'A' for 'U', 'W' for 'U').
    for text in (condensed, upper):
        for m in _BIC_EXACT.finditer(text):
            if is_valid_bic(m.group()) and m.group()[3] == 'U':
                return m.group(), "exact", 0

    seen, full_candidates, recovered_candidates = set(), [], []

    # 11-char alphanumeric windows starting with 4 letters
    for i in range(len(alnum_only) - 10):
        w = alnum_only[i: i + 11]
        if re.fullmatch(r"[A-Z]{4}[A-Z0-9]{7}", w) and w not in seen:
            seen.add(w); full_candidates.append(w)

    # Loose spaced pattern: e.g. "ATRU 816125 8"
    for m in _BIC_LOOSE.finditer(upper):
        chunk = _condense(m.group())
        if len(chunk) >= 11 and re.fullmatch(r"[A-Z]{4}", chunk[:4]):
            w = chunk[:11]
            if w not in seen:
                seen.add(w); full_candidates.append(w)

    # Windows with '?' uncertainty markers
    for i in range(len(wild_condensed) - 10):
        w = wild_condensed[i: i + 11]
        if re.fullmatch(r"[A-Z?]{4}", w[:4]) and "?" in w and w not in seen:
            seen.add(w); full_candidates.append(w)

    # Spacing-inferred wildcards
    for w in _spaced_wildcards(upper):
        if w not in seen:
            seen.add(w); full_candidates.append(w)
        # Block the no-wildcard condensed form from the 10-char prefix path.
        # Without this, e.g. 'CPWU804?188' would still let 'CPWU804188' slip
        # through as a 10-char prefix, compute check digit 2, and return the
        # wrong BIC (CPWU8041882) before the wildcard recovery even runs.
        no_wc = re.sub(r"[^A-Z0-9]", "", w)
        if no_wc not in seen:
            seen.add(no_wc)

    # Dashed format: XXXX-YYYYYY-Z
    dashed_wildcards, dashed_serial_hits, dashed_blocked = _parse_dashed_bic(
        re.sub(r'\s+', '-', upper)
    )
    for bf in dashed_blocked:
        seen.add(bf)
        if len(bf) == 11:
            # Only block the 10-char prefix if the blocked form is itself a valid BIC.
            # If invalid (e.g. size/type code digit bled into the check digit group),
            # the prefix must remain available for the check-digit recovery path.
            if is_valid_bic(bf):
                seen.add(bf[:10]); seen.add(bf[1:])
    full_candidates = [c for c in full_candidates if c not in dashed_blocked]
    # Non-U valid blocked forms (e.g. OCR misread U as A/W): remove from main pipeline
    # but keep for the owner brute-force pass that runs later.
    _non_u_blocked = [
        bf for bf in dashed_blocked
        if len(bf) == 11 and re.fullmatch(r"[A-Z]{4}\d{7}", bf)
        and is_valid_bic(bf) and bf[3] != 'U'
    ]
    for w in dashed_wildcards:
        if w not in seen:
            seen.add(w); full_candidates.append(w)
        stripped = re.sub(r'[^A-Z0-9]', '', w)
        if stripped not in seen:
            seen.add(stripped)
    if len(dashed_serial_hits) == 1:
        bic = dashed_serial_hits[0]
        if bic not in seen:
            seen.add(bic); recovered_candidates.append(bic)

    # 10-char windows: one character dropped entirely
    ambiguous_bics: list[str] = []
    for bic in (dashed_serial_hits if len(dashed_serial_hits) > 1 else []):
        if bic not in seen:
            seen.add(bic); ambiguous_bics.append(bic)
    for i in range(len(alnum_only) - 9):
        w = alnum_only[i: i + 10]
        if w in seen:
            continue
        seen.add(w)
        if re.fullmatch(r"[A-Z]{4}[0-9]{6}", w):
            computed = _compute_check(w)
            if i + 10 < len(alnum_only) and alnum_only[i + 10].isdigit():
                if int(alnum_only[i + 10]) != computed:
                    # Only skip if the competing 11-char read is itself a valid BIC.
                    # If it's not valid, the extra digit is likely a size/type code
                    # character that landed next to the check digit in the OCR output.
                    if is_valid_bic(w + alnum_only[i + 10]):
                        continue
            bic = w + str(computed)
            if bic not in seen:
                seen.add(bic); recovered_candidates.append(bic)
        elif re.fullmatch(r"[A-Z]{3}[A-Z0-9]{7}", w):
            hits = _insert_candidates(w)
            if len(hits) == 1:
                if hits[0] not in seen:
                    seen.add(hits[0]); recovered_candidates.append(hits[0])
            elif len(hits) > 1:
                for h in hits:
                    if h not in seen:
                        seen.add(h); ambiguous_bics.append(h)

    for c in full_candidates:
        if is_valid_bic(c) and c[3] == 'U':
            return c, "exact", 0
    # Non-U valid BICs fall through to owner brute-force to recover the U variant.

    for c in recovered_candidates:
        if is_valid_bic(c) and c[3] == 'U':
            return c, "recovered", 1
    for c in recovered_candidates:
        if is_valid_bic(c):
            return c, "recovered", 1

    wildcard_singles: list[str] = []
    for c in full_candidates:
        if "?" not in c:
            continue
        hits = _recover_wildcards_all(c)
        if len(hits) == 1:
            wildcard_singles.append(hits[0])
        elif len(hits) > 1:
            for h in hits:
                if h not in seen:
                    seen.add(h); ambiguous_bics.append(h)
    unique_singles = list(dict.fromkeys(wildcard_singles))
    if len(unique_singles) == 1:
        return unique_singles[0], "recovered", 1
    elif len(unique_singles) > 1:
        for h in unique_singles:
            if h not in seen:
                seen.add(h); ambiguous_bics.append(h)

    for c in full_candidates + _non_u_blocked:
        # Also include valid BICs whose 4th letter is not 'U' — likely OCR misread of 'U'
        if re.fullmatch(r"[A-Z]{4}\d{7}", c) and (not is_valid_bic(c) or c[3] != 'U'):
            hit = _brute_force_owner(c)
            if hit:
                cands, dist = hit
                # Prefer freight containers (4th letter = 'U', ISO 6346)
                u_cands = [h for h in cands if h[3] == 'U']
                effective = u_cands if u_cands else cands
                if len(effective) == 1:
                    return effective[0], "recovered", dist
                for h in effective:
                    if h not in seen:
                        seen.add(h); ambiguous_bics.append(h)

    if ambiguous_bics:
        # Prefer freight containers (4th letter = 'U', ISO 6346)
        u_first = [b for b in ambiguous_bics if len(b) >= 4 and b[3] == 'U']
        if len(u_first) == 1:
            return u_first[0], "recovered", 1
        return (u_first if u_first else ambiguous_bics), "brute_forced", 1

    for c in full_candidates + recovered_candidates:
        hit = _brute_force(c)
        if hit:
            candidates, dist = hit
            return candidates, "brute_forced", dist

    return None, "not_found", 0


# ─────────────────────────────────────────────────────────────────────────────
# Datalab API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_to(image_bytes: bytes, max_pixels: int) -> bytes:
    """Downscale image to at most max_pixels, always returning JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if img.width * img.height > max_pixels:
        scale = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _post_and_poll(endpoint: str, files: dict, data: dict) -> dict:
    """POST to a Datalab endpoint, poll until complete, return result dict."""
    if not DATALAB_API_KEY:
        raise RuntimeError("DATALAB_API_KEY environment variable is not set")
    headers = {"X-API-Key": DATALAB_API_KEY}
    # Retry POST with exponential backoff on 429 (DataLabs rate limit)
    delay = 5
    for attempt in range(6):
        r = requests.post(
            f"{_BASE}/{endpoint}",
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", delay))
            wait = max(retry_after, delay)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    body = r.json()
    check_url = body.get("request_check_url")
    if not check_url:
        raise RuntimeError(f"No request_check_url in response: {r.text[:300]}")

    deadline = time.time() + _POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(check_url, headers=headers, timeout=30)
        poll.raise_for_status()
        result = poll.json()
        status = result.get("status")
        if status == "complete":
            return result
        if status == "failed":
            raise RuntimeError(f"Datalab job failed: {result.get('error', 'unknown')}")
    raise TimeoutError(f"Datalab API did not complete within {_POLL_TIMEOUT}s")


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction schemas
# ─────────────────────────────────────────────────────────────────────────────

# Schema for a shipping container photo — asks for the BIC in our space-gap format.
# The model outputs spaces between visually separate groups and ? for covered chars,
# which feeds directly into extract_bic() for check-digit-based recovery.
_BIC_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "bic_raw": {
            "type": "string",
            "description": (
                "The BIC/ILU identification code on the shipping container. "
                "It consists of exactly: 4 owner letters + 6 serial digits + 1 check digit. "
                "The check digit is always a single digit (0-9) displayed alone in its own "
                "small separate box to the right of the serial. "
                "IMPORTANT — directly below the BIC there is often a 4-character ISO size/type "
                "code such as 22G1, 42G1, 45G1, 40HC, etc. This is NOT part of the BIC. "
                "Do not include it. The BIC ends with the check digit in its bordered box; "
                "everything on the line below is irrelevant. "
                "Output all visible BIC characters with a space between each visually separate "
                "group (a gap or covered section = one space). "
                "IMPORTANT: the BIC may be split across multiple lines on the container door — "
                "read ALL lines that form the BIC (owner prefix, then serial digits, then check digit) "
                "and combine them into one space-separated string. "
                "Use ? for any character that is covered, scratched out, or unclear. "
                "If the check digit looks like a letter instead of a digit, write ? for it. "
                "If the check digit box is obscured but the rest is readable, output the "
                "4 letters and 6 serial digits without a check digit. "
                "If no BIC code is visible at all, output exactly: none. "
                "Examples: 'MSCU 726354 1'  |  'TCLU 70?064 7'  |  'ATRU 81 125 8'  "
                "|  'PSSU 802397' (check digit box hidden)  "
                "|  'CPWU 804 18 8' (digits split across lines, read top-to-bottom)  "
                "|  'MOFU 077 1250 4' (serial continues on next line, stop before ISO size code)."
            ),
        },
    },
    "required": ["bic_raw"],
})


# Schema for a CMR (Convention on the Contract for the International Carriage of
# Goods by Road) consignment note. v5 descriptions + descriptive field names.
_CMR_SCHEMA = json.dumps({
    "type": "object",
    "title": "CMR_ExtractionSchema_v5",
    "description": "CMR consignment note. Fields may be printed, handwritten, or stamped. Check all three before returning null. Sender, carrier, and receiver are three different parties — do not merge or swap them.",
    "properties": {
        "box1_sender":                     {"type": "string", "description": "Sender's full name and address (box 1, top-left). Usually printed."},
        "box2_consignee":                  {"type": "string", "description": "Receiver's full name and address (box 2, below box 1). Usually handwritten."},
        "box3_place_of_delivery":          {"type": "string", "description": "Intended place of delivery (box 3, above box 4). Usually handwritten; often matches receiver's address but extract as written."},
        "box4_place_and_date_taking_over": {"type": "string", "description": "Place and date the carrier took over the goods (box 4). Usually handwritten."},
        "box5_documents_attached":         {"type": "string", "description": "Documents attached by sender (box 5). Usually handwritten, e.g. a packing list reference."},
        "box6_marks_and_numbers":          {"type": "string", "description": "Marks and numbers on packages (box 6). Often blank."},
        "box7_number_of_packages":         {"type": "string", "description": "Number of packages (box 7), e.g. '2 pallets'."},
        "box8_method_of_packing":          {"type": "string", "description": "Method of packing (box 8), e.g. 'pallets', 'cartons'."},
        "box9_nature_of_goods":            {"type": "string", "description": "Nature of goods (box 9), e.g. product name or category."},
        "box10_statistical_number":        {"type": "string", "description": "Statistical/HS number (box 10). Often blank."},
        "box11_gross_weight_kg":           {"type": "string", "description": "Gross weight in kg (box 11)."},
        "box12_volume_m3":                 {"type": "string", "description": "Volume in cubic meters (box 12). Often blank."},
        "box13_senders_instructions":      {"type": "string", "description": "Sender's instructions for customs/insurance (box 13). Often blank."},
        "box14_cash_on_delivery":          {"type": "string", "description": "Cash on delivery / reimbursement amount (box 15). Often blank."},
        "box15_carriage_charges":          {"type": "string", "description": "Carriage payment instructions and charges table (box 14/20). Often blank."},
        "box16_carrier":                   {"type": "string", "description": "Carrier's full name and address (box 16, right column). Usually printed or stamped."},
        "box17_successive_carriers":       {"type": "string", "description": "Successive/second carrier, if different from box 16 (box 17). Often blank or just vehicle plate numbers with no second carrier name."},
        "box18_carrier_reservations":      {"type": "string", "description": "Carrier's reservations/observations at pickup (box 18). Often blank."},
        "box19_special_agreements":        {"type": "string", "description": "Special agreements between sender and carrier (box 19). Often blank."},
        "box20_to_be_paid_by":             {"type": "string", "description": "Vehicle loading/departure date and time (box 22, near sender's signature). Usually handwritten."},
        "box21_established_at_date":       {"type": "string", "description": "Place and date the consignment note was issued (box 21). Usually handwritten."},
        "box22_sender_signature_stamp":    {"type": "string", "description": "Sender's signature and stamp (box 22). May be a stamp, a signature, or both. If a signature overlaps a stamp, also read any stamp text visible around or beside the signature strokes, not just the clearest line."},
        "box23_carrier_signature_stamp":   {"type": "string", "description": "Carrier's signature and stamp (box 23). May be a stamp, a signature, or both. Should match the carrier in box 16."},
        "box24_consignee_signature_stamp": {"type": "string", "description": "Receiver's signature and stamp confirming receipt (box 24). May be blank if delivery isn't confirmed yet, or stamp-only without a signature."},
    },
})

CMR_FIELDS = [
    "box1_sender", "box2_consignee", "box3_place_of_delivery",
    "box4_place_and_date_taking_over", "box5_documents_attached",
    "box6_marks_and_numbers", "box7_number_of_packages", "box8_method_of_packing",
    "box9_nature_of_goods", "box10_statistical_number", "box11_gross_weight_kg",
    "box12_volume_m3", "box13_senders_instructions", "box14_cash_on_delivery",
    "box15_carriage_charges", "box16_carrier", "box17_successive_carriers",
    "box18_carrier_reservations", "box19_special_agreements", "box20_to_be_paid_by",
    "box21_established_at_date", "box22_sender_signature_stamp",
    "box23_carrier_signature_stamp", "box24_consignee_signature_stamp",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def _clean_bic_raw(raw: str) -> str:
    """Post-process ocr_bic() output to fix two common model errors.

    1. Letter in check-digit position (e.g. 'Z' misread from a '7') → replace with '?'
       so wildcard recovery can fill it in.
    2. Single digit in check-digit position that makes the full BIC invalid → strip it
       so extract_bic() computes the correct check digit from the 10-char prefix.
       This catches the case where the model reads the first digit of the ISO type code
       (e.g. '4' from '45G1') instead of the actual check digit in the box above it.
    """
    tokens = raw.split()
    if not tokens or raw.strip().lower() == "none":
        return raw
    last = tokens[-1]
    # Case 1: single letter where a digit is expected → replace with '?'
    if re.fullmatch(r"[A-Z]", last):
        return " ".join(tokens[:-1]) + " ?"
    # Case 2: single digit, but the full condensed BIC fails the check digit test
    if re.fullmatch(r"\d", last):
        condensed = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(condensed) == 11 and re.fullmatch(r"[A-Z]{4}\d{7}", condensed):
            if not is_valid_bic(condensed):
                return " ".join(tokens[:-1])
    # Case 3: multi-digit last token that produced an invalid 11-char condensed form.
    # The OCR may have bundled the last serial digit with the check digit into one group
    # (e.g. "FCBU 86 092 17" where the real split is serial=86?092 check=7).
    # Strip the bundled token and re-attach only its last character as the check digit,
    # so that wildcard recovery can fill in the missing serial position.
    if re.fullmatch(r"\d{2,}", last):
        condensed = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(condensed) == 11 and re.fullmatch(r"[A-Z]{4}\d{7}", condensed):
            if not is_valid_bic(condensed):
                return " ".join(tokens[:-1]) + " " + last[-1]
    return raw


def check_image_content(image_bytes: bytes, reject_items: list[str]) -> tuple[bool, str]:
    """Return (rejected, reason). Asks DataLabs to check for forbidden objects on the document."""
    items_str = ", ".join(reject_items)
    schema = {
        "type": "object",
        "properties": {
            "forbidden_objects_present": {
                "type": "boolean",
                "description": (
                    f"True if any of these objects ({items_str}) are physically placed on top of "
                    "the document and blocking or covering its text. "
                    "False if the document surface is fully clear."
                ),
            },
            "forbidden_objects_found": {
                "type": "string",
                "description": f"Name the specific object ({items_str}) physically on the document, or null if none.",
            },
        },
        "required": ["forbidden_objects_present"],
    }
    image_bytes = _resize_to(image_bytes, 3840 * 2160)
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data  = {"page_schema": json.dumps(schema), "extraction_mode": "turbo"}
    result = _post_and_poll("extract", files, data)
    raw_json = result.get("extraction_schema_json") or "{}"
    try:
        extracted = json.loads(raw_json)
        if extracted.get("forbidden_objects_present"):
            found = extracted.get("forbidden_objects_found") or items_str
            return True, f"Image contains {found}. Please retake the photo without it."
        return False, ""
    except (json.JSONDecodeError, AttributeError):
        return False, ""


def ocr_bic(image_bytes: bytes) -> str:
    """Send a container image to Datalab and return a space-formatted BIC string.

    The returned string is fed directly into extract_bic() for check-digit recovery.
    Returns 'none' if no BIC code is visible.

    Uses extraction_mode='fast' ($6/1K pages) — sufficient for a single code read.
    """
    image_bytes = _resize_to(image_bytes, 1920 * 1440)
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data  = {"page_schema": _BIC_SCHEMA, "extraction_mode": "fast"}
    result = _post_and_poll("extract", files, data)
    raw_json = result.get("extraction_schema_json", "{}")
    try:
        extracted = json.loads(raw_json)
        value = (extracted.get("bic_raw") or "none").strip()
        return _clean_bic_raw(value)
    except (json.JSONDecodeError, AttributeError):
        return "none"


def ocr_cmr(image_bytes: bytes, quality: str = "balanced") -> dict:
    """Send a CMR document image to Datalab and return a dict of all 24 CMR fields.

    quality='balanced' for explicit DOCUMENT mode (best accuracy, ~60s).
    quality='fast'     for AUTO fallback (good enough, ~8s).
    """
    log.info(f"ocr_cmr: input {len(image_bytes)/1024:.0f}KB mode={quality}")
    image_bytes = _resize_to(image_bytes, 3840 * 2160)
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data  = {"page_schema": _CMR_SCHEMA, "extraction_mode": quality}
    result = _post_and_poll("extract", files, data)
    raw_json = result.get("extraction_schema_json") or "{}"
    log.info(f"ocr_cmr: got {sum(1 for v in json.loads(raw_json).values() if v)} fields")
    try:
        extracted = json.loads(raw_json)
    except (json.JSONDecodeError, AttributeError):
        extracted = {}
    return {k: extracted.get(k) or None for k in CMR_FIELDS}


def run_pipeline(image_bytes: bytes, mode: str = "auto", job_id: str = "") -> dict:
    """Full pipeline: extract BIC or CMR fields from an image.

    mode:
      'bic'  — container photo, only attempt BIC extraction
      'cmr'  — document photo, only attempt CMR extraction
      'auto' — try BIC first; if model says no container, fall back to CMR

    Return dict always has a 'mode' key ('bic' or 'cmr').
    BIC result keys: bic, candidates, confidence, edit_distance, ocr_raw
    CMR result keys: fields (dict of 24 CMR field values)
    """
    tag = f"[{job_id}] " if job_id else ""
    if mode == "cmr":
        t0 = time.time()
        fields = ocr_cmr(image_bytes, quality="turbo")
        log.info(f"{tag}datalab cmr done in {time.time()-t0:.1f}s")
        return {"mode": "cmr", "fields": fields}

    t0 = time.time()
    raw = ocr_bic(image_bytes)
    log.info(f"{tag}datalab bic done in {time.time()-t0:.1f}s — raw={raw!r}")

    if raw.strip().lower() != "none":
        bic_result, confidence, edit_distance = extract_bic(raw)
        if bic_result:
            if confidence == "brute_forced":
                return {
                    "mode": "bic",
                    "bic": None,
                    "candidates": bic_result,
                    "confidence": confidence,
                    "edit_distance": edit_distance,
                    "ocr_raw": raw,
                }
            return {
                "mode": "bic",
                "bic": bic_result,
                "candidates": [],
                "confidence": confidence,
                "edit_distance": edit_distance,
                "ocr_raw": raw,
            }
        # OCR returned something but extract_bic couldn't validate it — fall through

    if mode == "bic":
        return {
            "mode": "bic",
            "bic": None,
            "candidates": [],
            "confidence": "not_found",
            "edit_distance": 0,
            "ocr_raw": raw,
        }

    # auto mode: BIC not found, fall back to CMR
    t0 = time.time()
    fields = ocr_cmr(image_bytes, quality="balanced")
    log.info(f"{tag}datalab cmr fallback done in {time.time()-t0:.1f}s")
    return {"mode": "cmr", "fields": fields}
