# Ionization & Velocity Rules

## Related Knowledge

- Classification diagnostics (what to DO with ionization violations, fatal problems per class): see `kb/classification.md`
- Line rest wavelengths and doublet spacings: see `kb/lines.md`

## Systemic Redshift Anchoring Priority

Use the lowest-ionization LIKELY line to anchor systemic z. Priority from lowest to highest ionization:

| Priority | Lines | Ionization | Notes |
|----------|-------|------------|-------|
| 1 | Ca K/H_abs, G-band_abs, Mg I_abs, Na D_abs | Neutral | Stellar absorption — most reliable |
| 2 | [O II] 3727 | O⁺ (~13.6 eV) | Best emission anchor for ELG |
| 3 | [S II]a/b 6718/6733 | S⁺ | |
| 4 | [N II]a/b 6550/6585 | N⁺ | |
| 5 | Hα/Hβ/Hγ/Hδ/Hε | H (~13.6 eV) | |
| 6 | Mg II 2800 | Mg⁺ (~15.0 eV) | May show outflow blueshift |
| 7 | [O III]a/b 4960/5008 | O⁺⁺ (~35.1 eV) | Weakest anchor, often blueshifted |

**Excluded** (must NOT anchor systemic z): He II (He⁺, 54.4 eV), C III] (C⁺⁺, 47.9 eV), C IV (C⁺⁺⁺, 64.5 eV), [Ne V] (Ne⁺⁺⁺⁺, 97.1 eV), Lyα (1216). These high-ionization lines are routinely blueshifted by AGN outflows (hundreds of km/s). If only excluded lines are available, flag the redshift as potentially biased.

**Perceptual guidance**: Prioritize visual signal clarity over strict priority ordering. A visually dominant, unmistakable [O II] at Priority 2 is a better anchor than a marginal, barely-visible Ca K/H at Priority 1. The table is a tiebreaker, not a substitute for your visual judgment of which line is most convincingly detected.

## Ionization Consistency Rule

Higher-ionization lines of a given element imply the presence of lower-ionization lines of the same element. If a line requiring a higher ionization state is detected, the lower-ionization lines of that element MUST also be present (unless masked or at the spectrum edge).

The most important case for ELG classification:

| If detected | Then MUST also detect | Why |
|-------------|----------------------|-----|
| [O III] 4960/5008 (O⁺⁺, 35.1 eV) | [O II] 3727 (O⁺, 13.6 eV) | O⁺⁺ requires passing through O⁺ — an ionizing source strong enough to produce [O III] MUST produce abundant [O II] |

**[O III] without [O II] in clean, unmasked spectral regions is a physical contradiction.** Either:
- The [O III] identification is wrong (wrong z, wrong line ID, or OH skylines mimicking the doublet spacing), or
- [O II] is hiding in a masked/low-SNR region (check before rejecting).

This principle extends beyond oxygen to other elements: C IV (C⁺⁺⁺) requires C III] (C⁺⁺); [Ne V] (Ne⁺⁺⁺⁺) requires [Ne IV] and lower neon lines. Always check for lower-ionization counterparts of the same element before accepting a high-ionization line identification.

## Outflow Blueshift Rule

High-ionization lines blueshifted relative to low-ionization by 0–1000 km/s is physically normal.
Velocity offset: Δv = (z_high − z_low) / (1 + z_low) × c. Negative Δv = blueshift.
If a high-ionization line gives a LOWER z than a low-ionization line, suspect misidentification (not outflow reversal).

## Width Mismatch Policy

- A `narrow` feature matching a `broad` line (Lyα, C IV, C III], Mg II) → flag, do not use for systemic z
- A `broad` feature matching a `narrow` line ([O II], [O III], [N II], [S II]) → flag, suspect spurious CWT feature
- Balmer lines (Hα–Hδ) and He II are `both` class → width checks do not apply
- Flag mismatches but do not veto an entire hypothesis on one width mismatch alone
- Use your visual judgment: a feature that **looks** broad (spanning tens of pixels, smooth wings) carries more weight than the exact FWHM value
