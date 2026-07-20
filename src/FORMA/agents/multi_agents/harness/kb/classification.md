# Classification Diagnostics

## Related Knowledge

- Ionization physics (anchoring priorities, consistency rules, why [O III] requires [O II]): see `kb/ionization.md`
- Line rest wavelengths and doublet spacings: see `kb/lines.md`
- Emission-absorption composite profiles (Mg II, Hα, Hβ): see `kb/composite_profile.md`

## ELG (Emission Line Galaxy)

**Expected features**: Strong narrow emission lines ([O II], Hβ, [O III], Hα). Weak or absent absorption.

**Best systemic anchors**: [O II] 3727 (Priority 2), Hα (Priority 5).

**Ca K/H absorption**: Weak or absent is EXPECTED in ELGs — young stellar populations have weak metal absorption. Do NOT exclude an ELG hypothesis because Ca K/H features are MARGINAL, NOT_FOUND, have wrong separation, or show inverted K/H ratio. Ca K/H is a PRIMARY diagnostic for LRG/BGS only; for ELG it carries zero exclusion weight. An ELG hypothesis with perfect emission-line consistency but failed Ca K/H should NOT be penalized — the emission lines drive the classification. In emission-line dominated objects, anchor systemic z on [O II] (Priority 2) instead of Ca K/H.

**Fatal problems**:
- **[O III] without [O II]** — ionization inconsistency. See `kb/ionization.md` for the physics: [O III] (O⁺⁺) cannot exist without [O II] (O⁺) in a clean, unmasked region.
- **[O III] doublet spacing wrong** — both components present but separation doesn't match 47.9×(1+z) Å.
- **[O III] doublet orphan** — only one component detected (typically [O III]b) while the other is NOT_FOUND in a clean, unmasked region. If [O III]b is bright enough to be KEEP, [O III]a MUST be detectable. A missing partner means the feature is likely NOT [O III] — treat with the same weight as wrong spacing.

**Balmer lines (Hβ, Hγ, Hδ)**: Frequently weak or undetectable in ELGs due to moderate SFR, dust extinction, or low SNR. A missing Balmer line is NOT fatal — note as a caveat only.

**Dn4000**: Typically < 1.4 (young stellar population). A reference continuum metric, NOT a classification criterion.

## LRG/BGS (Luminous Red Galaxy / Bright Galaxy Sample)

**Expected features**: Strong stellar absorption (Ca K/H, G-band, Mg I, Na D). Weak or no emission lines.

**Best systemic anchors**: Ca K/H_abs, G-band_abs (Priority 1).

**Ca K/H**: MUST appear together with Ca K deeper than Ca H. Missing partner → hard exclusion.

**4000 Å break**: Should be present at 4000×(1+z) Å. Dn4000 > 1.6 is characteristic of old stellar populations but is a REFERENCE metric only — a low Dn4000 does NOT disqualify LRG/BGS classification. Note any Dn4000 inconsistency in caveats.

**Fatal problems**: Ca K/H missing (one present without the other AND no 4000 Å break observed), broad emission lines with FWHM > 2000 km/s detected by AGN line checks.

## QSO (Quasar / AGN — Type 1 and Type 2)

**Type 1 (broad-line) QSO**: Broad emission lines (Lyα, C IV, C III], Mg II) with FWHM > 2000 km/s. Narrow forbidden lines may also be present.

**Type 2 (narrow-line / obscured) QSO**: Narrow emission lines only (no broad BLR), but with at least one high-ionization forbidden line — [Ne V] 3426. The absence of broad lines is EXPECTED in Type 2 — it is NOT a fatal problem. The key diagnostic is the presence of [Ne V], which requires AGN-level ionization (97.1 eV) that cannot be produced by stellar photoionization.

**Best systemic anchors**: Mg II 2800 (Priority 6), [O II] (Priority 2) if narrow component visible.

**Amplitude ordering**: In typical QSO spectra, Lyα is the strongest broad line, followed by C IV, then C III], then Mg II. Significant deviations from this ordering suggest the line identifications should be re-examined — particularly if Mg II is the brightest claimed broad line. Use your perceptual judgment: a visually dominant broad line assigned to Mg II when Lyα and C IV are weak or absent is suspicious.

**Lyα multi-peak fragmentation**: IGM absorption can split broad Lyα into 2–3 apparent peaks along the line of sight. Each fragment may match Lyα at slightly different implied z. This is physically normal — the true Lyα center lies among the detected fragments. Multiple narrow/intermediate Lyα matches at nearby wavelengths can still support a QSO hypothesis.

**Fatal problems for Type 1 QSO**: All claimed broad lines are narrow (FWHM < 1500 km/s). Any of Lyα, C IV, C III], or Mg II missing at its predicted position (within the observed wavelength range — NOT MASKED) is fatal. A single missing broad line among these four is sufficient to reject the QSO classification. A spectroscopically convincing broad line (spanning tens of pixels, smooth wings) carries more weight than the exact FWHM value.

**Fatal problems for Type 2 QSO**: [Ne V] is NOT visually convincing, AND no broad lines are present. A Type 2 QSO without [Ne V] is indistinguishable from a star-forming Galaxy — do NOT classify as QSO in this case. If [Ne V] IS visually convincing, Type 2 QSO is a valid classification even with zero broad lines.

## Star

**Expected features**: Broad absorption lines, no emission lines, may show 4000 Å break.

**Distinction from LRG**: Star spectra show broader and deeper Balmer absorption than LRGs.

**If suspected**: Flag as UNKNOWN rather than committing if evidence is marginal.

## [Ne V] as AGN Indicator

[Ne V] (3426 Å) is a high-ionization forbidden line almost never present in non-AGN objects. However, it is a weak line easily mimicked by noise. Before using [Ne V] as AGN evidence:
1. Read the spectrum ±50 Å around the predicted observed wavelength.
2. Verify the feature is a genuine emission peak rising clearly above the local continuum — not a continuum wiggle or noise spike.
3. Weigh against Galaxy features: if Galaxy indicators (Ca K/H doublet, narrow emission lines with correct spacing) are clear and self-consistent while the [Ne V] feature is marginal, default to Galaxy.
4. [Ne V] CAN independently support QSO classification — but only if the peak is visually convincing, not merely a CWT detection at the noise limit.

## Mg II Emission vs Absorption Coexistence

Mg II (2800 Å) can appear as both broad emission (QSO BLR) and narrow absorption (ISM). When BOTH are claimed near the same observed wavelength, they may form a single emission–absorption composite profile. **For the full diagnostic criteria — the morphological "M" test, center consistency, wing broadness, symmetry, and spike–valley–spike detection — see `kb/composite_profile.md`.**

Key principle: a genuine composite profile supports both claims as a linked physical system. Spike–valley–spike or asymmetric noise does NOT support either claim. In ambiguous cases, default to the absorption interpretation — Mg II ISM absorption is far more common than Mg II BLR emission in non-QSO objects.

This composite-profile logic also applies to Hα + Hα_abs and Hβ + Hβ_abs systems.

## Broad Lines in Non-AGN Galaxies

In ELG/LRG, genuine broad emission lines (Lyα, C IV, C III], Mg II) do not appear. If CWT labels a feature as `broad` matching these lines in a galaxy hypothesis, suspect CWT artifact — overfitting of the continuum, fragmentation of a narrow line by noise, or a spurious wide Gaussian from poor baseline fit. Flag such cases — do not accept a broad classification in a galaxy hypothesis without noting the caveat.

## Final Classification Mapping

Sub-type labels (ELG, LRG/BGS, Host Galaxy dominated AGN) are used for internal reasoning and physical diagnostics. The final JSON verdict classification must map to ONLY these top-level categories:
- All galaxy sub-types (ELG, LRG, BGS, LRG/BGS, composite, star-forming) → `Galaxy`
- Host Galaxy dominated AGN, Pure QSO, Type 1 AGN → `QSO`
- Unclassifiable → `Unknown`

## Cross-Type Evidence Weight

- **LRG vs LRG**: Absorption lines (Ca K/H, G-band, Mg I, Na D) are primary discriminators. Emission lines secondary.
- **ELG vs ELG**: Emission lines ([O II], Hβ, [O III], Hα) are primary discriminators. Absorption lines secondary.
- **Cross-type** (LRG vs ELG vs QSO): Neither evidence type inherently more trustworthy. Judge each hypothesis on its own internal physical consistency, then compare completeness and coherence of line inventories. The deciding factor is physical diagnostics (line ratios, ionization consistency, continuum features), not line count.
