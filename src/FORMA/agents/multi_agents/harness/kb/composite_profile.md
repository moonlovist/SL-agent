# Emission–Absorption Composite Profile

## Motivation

CWT and line-finding algorithms frequently identify emission peaks and absorption troughs independently. When an absorption feature is superposed on an emission line of the same species, the resulting profile may be fragmented into a central trough identified as an absorption line, and one or more adjacent maxima identified as emission lines — when in reality they are two halves of a single physical system.

Whenever an emission line and an absorption line of the same species are detected at nearly the same wavelength, evaluate them as a possible composite profile.

## Physical Interpretation

A composite profile occurs when absorbing material along the line of sight removes flux near the emission line center, producing an emission feature with a central depression. The absorption does NOT necessarily represent an independent spectral feature — it may merely modify the shape of a single underlying emission line.

## Morphological Signature

A genuine emission–absorption composite profile forms a broad "M" shape in the spectrum: two emission wings flanking a central absorption trough, with smooth flux continuity between all components and approximate symmetry around a common center.

## Diagnostic Criteria

### Criterion 1: Common Central Wavelength
The absorption trough should lie near the center of the emission structure: λ_abs ≈ (λ_left + λ_right) / 2. Large offsets (> half the emission FWHM) weaken the composite interpretation.

### Criterion 2: Broad and Smooth Wings
The two emission wings should be broad, smooth, and continuous — not narrow spikes. Genuine composites preserve the morphology of the parent emission line. Spike–valley–spike patterns (two narrow peaks flanking a trough) are commonly produced by noise fluctuations and do NOT support a physical composite.

### Criterion 3: Approximate Symmetry
The left and right wings should have comparable width, curvature, and amplitude. Severe asymmetry often indicates noise contamination, incorrect line identification, or blending with another transition.

### Criterion 4: Continuum Connectivity
The wings should merge naturally into the surrounding continuum. Profiles composed of isolated spikes with no broad structure are unreliable.

## Warning Signs

**Noise-Induced False Composite**: Multiple local extrema without a coherent broad profile. Both emission and absorption detections are likely spurious.

**Spike–Valley–Spike**: Two narrow peaks surrounding a narrow trough. Commonly produced by noise — does NOT support a physical composite system. This is the most important false-positive pattern to recognize.

**Strongly Asymmetric**: Only one side exhibits a broad emission wing. May indicate line blending with another species, incorrect absorption assignment, or noise on one side of the emission profile. Should not automatically be interpreted as a composite.

## Recommended Agent Behavior

When an emission and absorption line of the same species are detected:

1. Search for a shared morphological structure across both detections.
2. Estimate the center of the combined profile.
3. Evaluate center consistency, wing broadness and smoothness, symmetry, and continuum connectivity.
4. If a coherent composite "M" profile is present: treat the detections as a **single physical system** and increase confidence in the line identification.
5. If the profile is dominated by spikes, asymmetry, or noise: reduce confidence for both detections. Default to treating them as independent (low-confidence) rather than linked.

## Examples

### Mg II + Mg II_abs (2800 Å rest)

The most common and important composite case. Mg II can appear as both broad emission (QSO BLR, FWHM > 2000 km/s) and narrow absorption (ISM, FWHM < 1000 km/s). When BOTH are claimed near the same observed wavelength:

**Center coincidence check**: If |λ_em − λ_abs| > max(FWHM_em, FWHM_abs), the two features are physically unrelated — one is a misidentification.

**Absorption-dominant false emission**: If the CWT feature at the predicted Mg II position is NARROW and in ABSORPTION (FWHM < 1000 km/s), the nearby broad "Mg II emission" is likely a CWT artifact from overfitting the continuum between absorption troughs, broad noise on the absorption wings, or poor baseline subtraction.

**Default to absorption in ambiguous cases**: Mg II ISM absorption is ubiquitous; Mg II BLR emission requires a genuine QSO. The emission claim requires POSITIVE morphological evidence: clearly broad profile (FWHM > 2000 km/s), clearly distinct from the absorption feature, and supported by at least one other AGN indicator ([Ne V], C III], C IV).

**Composite verdict**: A genuine broad, symmetric "M" shape with smooth wings supports both the emission AND absorption claims as a single linked physical system. Spike–valley–spike or asymmetric noise does NOT support either claim. Default to Galaxy classification if the only AGN evidence is a dubious Mg II composite.

### Hα + Hα_abs (6564.6 Å rest)

Broad or narrow Hα emission with stellar or interstellar absorption near the center. Common in galaxy spectra containing mixed stellar and nebular components. The same criteria and morphological test apply.

### Hβ + Hβ_abs (4862.7 Å rest)

Common in post-starburst galaxies and stellar-dominated systems. The underlying Balmer absorption may partially remove the nebular emission core, producing a composite profile. Same criteria as Mg II.

## General Principle

A line should not be classified solely from individual local extrema. Whenever an emission feature and an absorption feature of the same transition coexist at nearly identical wavelengths, **the morphology of the entire profile takes precedence over the individual detections.**
