def safe_to_bool(value):
    """专门处理true/True相关值的转换"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ['true', '1', 't', 'yes', 'y']
    return bool(value)

# ===========================================================
# Wavelength Band Overlap
# ===========================================================

def find_overlap_regions(band_names, band_wavelengths):
    """
    找出所有波段之间的重叠区域

    参数:
    band_names: 波段名称列表
    band_wavelengths: 波段波长范围列表，每个元素为[start, end]

    返回:
    重叠区域的字典
    """
    result = {}
    n = len(band_names)

    # 遍历所有可能的两两组合
    for i in range(n):
        for j in range(i + 1, n):
            # 获取两个波段的波长范围
            start1, end1 = band_wavelengths[i]
            start2, end2 = band_wavelengths[j]

            # 计算重叠区域
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)

            # 如果存在重叠区域
            if overlap_start < overlap_end:
                overlap_name = f"{band_names[i]}-{band_names[j]}"
                result[overlap_name] = [overlap_start, overlap_end]

    return result


# ===========================================================
# Line name normalisation
# ===========================================================

_LINE_ALIASES: dict[str, tuple[str, float]] = {
    # ── Emission lines ─────────────────────────────────────────
    "O [II]":            ("[O II] 3727",    3727.0),
    "[O II]":             ("[O II] 3727",    3727.0),
    "[O II] 3727":        ("[O II] 3727",    3727.0),
    "O[II]":              ("[O II] 3727",    3727.0),
    "Ne [V]":             ("[Ne V] 3426",      3426.0),
    "Ne V":               ("[Ne V] 3426",      3426.0),
    "Ne[V]":              ("[Ne V] 3426",      3426.0),
    "O [III]a":           ("[O III]a 4960.3", 4960.3),
    "[O III]a":           ("[O III]a 4960.3", 4960.3),
    "O [III]b":           ("[O III]b 5008.2", 5008.2),
    "[O III]b":           ("[O III]b 5008.2", 5008.2),
    "N[II]a":             ("[N II]a 6549.9", 6549.9),
    "N [II]a":            ("[N II]a 6549.9", 6549.9),
    "[N II]a":            ("[N II]a 6549.9", 6549.9),
    "N[II]b":             ("[N II]b 6585.3", 6585.3),
    "N [II]b":            ("[N II]b 6585.3", 6585.3),
    "[N II]b":            ("[N II]b 6585.3", 6585.3),
    "S[II]a":             ("[S II]a 6718.3", 6718.3),
    "S [II]a":            ("[S II]a 6718.3", 6718.3),
    "[S II]a":            ("[S II]a 6718.3", 6718.3),
    "S[II]b":             ("[S II]b 6732.7", 6732.7),
    "S [II]b":            ("[S II]b 6732.7", 6732.7),
    "[S II]b":            ("[S II]b 6732.7", 6732.7),
    "Mg II":              ("Mg II 2800",     2800.0),
    "Mg II 2800":         ("Mg II 2800",     2800.0),
    "C IV":               ("C IV 1549",      1549.0),
    "C III]":             ("C III] 1909",    1909.0),
    "He II":              ("He II 1640.4",   1640.4),
    "Lyα":                ("Lyα 1216",       1216.0),
    "Hα":                 ("Hα 6564.6",      6564.6),
    "Hβ":                 ("Hβ 4862.7",      4862.7),
    "Hγ":                 ("Hγ 4341.7",      4341.7),
    "Hγ 4341.7":          ("Hγ 4341.7",      4341.7),
    "Hδ":                 ("Hδ 4102.9",      4102.9),
    "Hε":                 ("Hε 3970.1",      3970.1),
    "Hε 3970":            ("Hε 3970.1",      3970.1),
    # ── Absorption lines ───────────────────────────────────────
    "Ca K":               ("Ca K 3934.8_abs",  3934.8),
    "Ca K_abs":           ("Ca K 3934.8_abs",  3934.8),
    "Ca H":               ("Ca H 3969.6_abs",  3969.6),
    "Ca H_abs":           ("Ca H 3969.6_abs",  3969.6),
    "Na D_abs":           ("Na D 5895_abs",    5895.0),
    "Na D":               ("Na D 5895_abs",    5895.0),
    "G-band_abs":         ("G-band 4305.6_abs", 4305.6),
    "G-band":             ("G-band 4305.6_abs", 4305.6),
    "Mg_abs":             ("Mg I 5176.7_abs",  5176.7),
    "Mg I_abs":           ("Mg I 5176.7_abs",  5176.7),
    "Hδ_abs":             ("Hδ 4102.9_abs",    4102.9),
    "Hγ_abs":             ("Hγ 4341.7_abs",    4341.7),
    "Hβ_abs":             ("Hβ 4862.7_abs",    4862.7),
    "Hα_abs":             ("Hα 6564.6_abs",    6564.6),
    "Hε_abs":             ("Hε 3970.1_abs",    3970.1),
    "Mg II_abs":          ("Mg II 2800_abs",   2800.0),
    "CaT2_abs":           ("CaT2 8544.4_abs",  8544.4),
    "CaT3_abs":           ("CaT3 8662.0_abs",  8662.0),
}


def normalize_line_name(raw: str) -> str:
    """Normalise a spectral line name to canonical form.

    Handles common LLM-generated variants (bracket ordering, spacing)
    then looks up the canonical name via alias table.

    Returns the canonical name (e.g. ``[O II] 3727``) or the cleaned
    string if no mapping is found.
    """
    import re

    # ── Step 1: bracket normalisation (regex) ──
    cleaned = re.sub(r'([A-Za-z]+)\s+\[([IVab]+)\]', r'\1[\2]', raw)
    _forbidden_map = {
        "O[II]": "[O II]", "N[II]a": "[N II]a", "N[II]b": "[N II]b",
        "S[II]a": "[S II]a", "S[II]b": "[S II]b",
        "O[III]a": "[O III]a", "O[III]b": "[O III]b", "Ne[V]": "[Ne V]",
        "O II": "[O II]", "O IIIa": "[O III]a", "O IIIb": "[O III]b",
        "N IIa": "[N II]a", "N IIb": "[N II]b",
        "S IIa": "[S II]a", "S IIb": "[S II]b", "Ne V": "[Ne V]",
    }
    if cleaned in _forbidden_map:
        cleaned = _forbidden_map[cleaned]
    if cleaned == "Mg_abs":
        cleaned = "Mg I_abs"

    # ── Step 2: alias lookup ──
    if cleaned in _LINE_ALIASES:
        return _LINE_ALIASES[cleaned][0]
    # Try stripping trailing "(Balmer ...)" annotations
    stripped = re.sub(r'\s*\(.*\)', '', cleaned).strip()
    if stripped in _LINE_ALIASES:
        return _LINE_ALIASES[stripped][0]
    return cleaned


# ===========================================================
# Generic helpers
# ===========================================================

def parse_csv_float(val: str) -> float | None:
    """Parse a CSV string to float, returning None on empty/invalid."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def average_flux_by_wavelength(wavelength, flux):
    """对同一波长的 flux 进行简单平均。

    Returns (unique_wavelength, mean_flux).
    """
    import pandas as pd
    df = pd.DataFrame({'wavelength': wavelength, 'flux': flux})
    mean_flux = df.groupby('wavelength', group_keys=False)['flux'].mean()
    return mean_flux.index.values, mean_flux.values
