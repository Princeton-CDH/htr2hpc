# Python Dependencies Review — htr2hpc / eScriptorium

**Date:** 2026-08-10
**Author:** Hao Tan / Princeton CDH
**Scope:** eScriptorium dev-0.14.2 (current), v1.0.0, 26.04 — plus htr2hpc (develop branch) and Kraken
**Staging environment verified:** `cdh-test-htr1.lib.princeton.edu` — web venv: kraken 5.3.0, htr2hpc 0.7.0, Django 4.2.30

______________________________________________________________________

## Background

This review was conducted to support the decision of whether to upgrade the Princeton eScriptorium deployment from the current base (dev-0.14.2) to either v1.0.0 or 26.04. The goal is to identify all Python dependencies across relevant components, understand what changes between versions, and propose how to manage dependencies for each runtime environment.

______________________________________________________________________

## 1. Raw Dependency Inventory

### 1.1 eScriptorium `requirements.txt` — three versions compared

Data sources (fetched 2026-08-10 via GitLab API):

- dev-0.14.2: `gitlab.com/scripta/escriptorium` tag `dev-0.14.2`
- v1.0.0: `gitlab.com/scripta/escriptorium` tag `v1.0.0`
- 26.04: `gitlab.com/scripta/escriptorium` tag `26.04`

| Package | dev-0.14.2 (current base) | v1.0.0 | 26.04 |
|---|---|---|---|
| albumentations | ✓ (unpinned) | ✓ (unpinned) | ✓ (unpinned) |
| beautifulsoup4 | ~=4.7.1 | ~=4.7.1 | ~=4.7.1 |
| bleach | ~=5.0.1 | ~=5.0.1 | ~=5.0.1 |
| celery | ~=5.3.6 | ~=5.3.6 | ~=5.3.6 |
| channels | ~=3.0.4 | ~=3.0.4 | ~=3.0.4 |
| channels-redis | ~=3.4.0 | ~=3.4.0 | ~=3.4.0 |
| daphne | ~=3.0.2 | ~=3.0.2 | ~=3.0.2 |
| **dfine-kraken** | — | — | ~=0.4.1 ✨ |
| Django | ~=4.2.13 | ~=4.2.13 | ~=4.2.13 |
| django-cleanup | ~=8.1.0 | ~=8.1.0 | ~=8.1.0 |
| django-country-and-language-flags | ✓ | ✓ | ✓ |
| django-filter | ✓ | ✓ | ✓ |
| django-ordered-model | ~=3.7 | ~=3.7 | ~=3.7 |
| django-prometheus | ~=2.3.1 | ~=2.3.1 | ~=2.3.1 |
| django-redis | ~=5.4.0 | ~=5.4.0 | ~=5.4.0 |
| django-simple-captcha | ~=0.6.0 | ~=0.6.0 | ~=0.6.0 |
| **django-solo** | — | >=1.1 ✨ | >=1.1 ✨ |
| djangorestframework | ~=3.15.1 | ~=3.15.1 | ~=3.15.1 |
| drf-nested-routers | ~=0.91 | ~=0.91 | ~=0.91 |
| easy-thumbnails | ~=2.8.1 | ~=2.8.1 | ~=2.8.1 |
| elasticsearch | ~=7.17.0 | ~=7.17.0 | ~=7.17.0 |
| **kraken** | >=5.2.9,~=5.2 | >=6.0.0,~=6.0 | ~=7.0.1 |
| oitei | ~=1.1.1 | ~=1.1.1 | ~=1.1.1 |
| passim (git) | @v2.0.1 | @v2.0.1 | @v2.0.1 |
| Pillow | >=5.4.1 | >=5.4.1 | >=5.4.1 |
| psutil | ✓ | ✓ | ✓ |
| psycopg2-binary | ✓ | ✓ | ✓ |
| **pyspark** | — | — | ==4.1.1 ⚠️ |
| pyvips | ~=2.1.12 | ~=2.1.12 | ~=2.1.12 |
| redis | ~=4.4.4 | ~=4.4.4 | ~=4.4.4 |
| requests | >=2.21 | >=2.21 | >=2.21 |
| sentry-sdk | >=1.10 | >=1.10 | >=1.10 |
| uWSGI | ~=2.0.20 | ~=2.0.20 | ~=2.0.20 |

**Legend:** ✨ = added, ⚠️ = requires attention

**Key changes across versions:**

- `django-webpack-loader` and `drf-spectacular`: not present in any of the three upstream tags; these appear in some Princeton-local branches only
- `django-solo` added in v1.0.0+ (enables Celery batch size configuration via Django admin)
- Kraken constraint bumped: `>=5.2.9,~=5.2` → `>=6.0.0,~=6.0` (v1.0.0) → `~=7.0.1` (26.04)
- 26.04 adds `dfine-kraken~=0.4.1` (D-FINE segmentation model support) and `pyspark==4.1.1` (see Section 3.2)

### 1.2 eScriptorium `requirements-dev.txt` — all three versions identical

```
django-debug-toolbar~=3.2.1
django-extensions~=3.1.0
pre-commit
```

No changes across dev-0.14.2, v1.0.0, and 26.04.

### 1.3 htr2hpc `pyproject.toml` (develop branch)

**Runtime dependencies:**

```toml
[project]
dependencies = [
  "pucas>=0.11",
  "psutil",
  "requests",
  "kraken",           # unpinned — no version constraint (see Note 1)
  "humanize",
  "simple_slurm",
  "tqdm",
  "intspan",
  "fabric",
  "django~=4.2",      # pinned to match eScriptorium
]
```

**Dev/test dependencies:**

```toml
[dependency-groups]
dev = [
  "pre-commit",
  "pytest>=8.3",
  "pytest-django>=4.9",
  "pytest-cov>=6.1",
  "pytest-mock>=3.14",
  "coverage[toml]>=7.6",
  "responses>=0.25",
  {include-group = "docs"},
]
docs = [
  "sphinx>=7.2",
  "alabaster",
  "django~=4.2",
  "myst-parser",
]
```

### 1.4 Kraken dependency comparison (PyPI)

Values are from the specific release that will actually be installed by each eScriptorium version's constraint. All three columns use the same `requires_dist` source format.

| Dependency | kraken 5.3.0 (resolves from dev-0.14.2 `>=5.2.9,~=5.2`) | kraken 6.0.3 (HPC install; resolves from v1.0.0 `>=6.0.0,~=6.0`) | kraken 7.0.3 (resolves from 26.04 `~=7.0.1`) |
|---|---|---|---|
| requires-python | >=3.9,\<3.13 | >=3.9,\<3.13 | >=3.10,\<3.14 |
| torch | ~=2.4.0 | >=2.4.0,\<=2.9 | >=2.4.0,\<=2.12 |
| lightning | ~=2.4.0 | ~=2.4.0 | ==2.6.1 |
| numpy | ~=2.0.0 | ~=2.0.0 | (unpinned) |
| scikit-image | ~=0.24.0 | ~=0.24.0 | ~=0.25.2 |
| scikit-learn | ~=1.5.0 | ~=1.5.0 | ~=1.7.2 |
| scipy | ~=1.13.0 | ~=1.13.0 | ~=1.15.3 |
| shapely | >=2.0.6,~=2.0.6 | >=2.0.6,~=2.0.6 | ~=2.1.2 |
| coremltools | ~=8.1 | ~=8.1 | ~=9.0 |
| python-bidi | ~=0.6.0 | ~=0.6.0 | vendored (see below) |
| htrmopo | — | >=0.3,~=0.3 ✨ | >=0.5,~=0.5 |
| safetensors | — | — | ~=0.7.0 ✨ |
| packaging | — | — | ✓ ✨ |
| iso639-lang | — | ✓ ✨ | ✓ |
| platformdirs | — | ✓ ✨ | ✓ |

**Key observation: the 5→6 dependency jump already happened inside 5.x.** numpy 2.x, shapely 2.x, coremltools 8.x, torch ~=2.4, and scikit-\* were all adopted in kraken 5.3.0. The 5→6 transition at the dependency level adds only `htrmopo`, `iso639-lang`, and `platformdirs`. The significant library upgrades are not a risk of the 5→6 transition — they are already the current state.

**python-bidi in 7.0.3 (vendored):** kraken 7.0.3 drops `python-bidi` from `requires_dist` intentionally — the library has been vendored into `kraken/lib/bidi/` as an internal subpackage. Imports changed from `import bidi.algorithm as bd` (external, 6.x) to `from kraken.lib.bidi import get_display_map` (internal, 7.0.3+). A clean environment will not hit `ImportError`; no workaround is needed.

______________________________________________________________________

## 2. Dependencies by Environment

### 2.1 HPC Training Environment

**Purpose:** submit and run Kraken training jobs on Della/HPC. No web server, no database, no frontend.

**Actual HPC installation** (`src/htr2hpc/train/user_setup.sh`): conda env with Python 3.11, then `pip install kraken==6.0.3`, then `pip install git+https://github.com/Princeton-CDH/htr2hpc.git@develop`. This means the HPC environment is already running **kraken 6.0.3**, while the web application is on dev-0.14.2 (kraken ~=5.2). See Findings 3.4 and 3.5.

**Required:**

| Package | Source | Code evidence | Notes |
|---|---|---|---|
| kraken | htr2hpc / eScriptorium | `train/data.py:8–13` | core training engine |
| torch, lightning, scikit-\*, numpy, scipy | kraken transitive deps | — | GPU/CUDA-dependent |
| simple_slurm | htr2hpc | `train/slurm.py:7` | Slurm job submission; used only in `train/slurm.py`, not in `tasks.py` |
| tqdm | htr2hpc | `train/run.py:16`, `train/data.py:14`, `train/rm_models.py:23` | progress bars in training scripts |
| intspan | htr2hpc | `train/run.py:14` (type hint in `TrainingManager`); also `tasks.py:13` | used on both sides; type hint in HPC-side dataclass |
| humanize | htr2hpc | `api_client.py:11` | file size display in API client, used by HPC training path |
| requests | htr2hpc | `api_client.py:12`, `train/run.py:13` | HTTP calls to eScriptorium API from HPC |
| **django** | htr2hpc (indirect) | `api_client.py:13` | `django.utils.text.slugify` only — see Finding 3.6 |
| pucas | htr2hpc (transitive) | — | pulled in as htr2hpc install dep; not called in any HPC code path |

**Not on HPC — web side only:**

| Package | Code evidence | Reason |
|---|---|---|
| fabric | `tasks.py:12` | imported only in `tasks.py` (Celery tasks); used to SSH *from* the web server *to* HPC, not run on HPC itself |
| psutil | `context_processors.py:4` | Django context processor for web VM memory display; no usage in `train/` |

**Not needed at all on HPC:** DRF, Celery, Redis, PostgreSQL, uWSGI, daphne, channels, pyvips, elasticsearch, passim, oitei, dfine-kraken, pyspark.

**Python version constraint:** kraken 5.3.0 and 6.x require `>=3.9,<3.13`; kraken 7.x requires `>=3.10,<3.14`. HPC runs Python 3.11, which satisfies all three. Note the upper bound: none of the three supports Python 3.13+.

### 2.2 Web Application Deployment (Production)

**Purpose:** full eScriptorium instance at Princeton with htr2hpc integration.

**Required:** all of `requirements.txt` for the target eScriptorium version, plus:

| Package | Source | Notes |
|---|---|---|
| htr2hpc | htr2hpc | installed via ansible `python_extra_packages` |
| pucas>=0.11 | htr2hpc | CAS authentication |

**Version-specific additions:**

- v1.0.0: adds `django-solo`; drops `django-webpack-loader` and `drf-spectacular` (neither was present in any upstream tag; these appeared only in Princeton-local branches)
- 26.04: additionally adds `dfine-kraken~=0.4.1`, `pyspark==4.1.1`

**Deployment note:** a single shared virtualenv (`django_venv_path`) installs eScriptorium `requirements.txt` first, then htr2hpc on top via ansible's `python_extra_packages`. Because htr2hpc's `kraken` dep is currently unpinned, pip will keep whichever version eScriptorium already installed rather than pulling a different one — but this relies on pip's "already satisfied" behaviour and could change if ansible invokes pip with `--upgrade` or resolves all packages together. The exact ansible pip invocation should be confirmed before relying on this ordering guarantee.

### 2.3 Local Development / Testing

**Purpose:** develop and test htr2hpc code locally, without a full eScriptorium installation.

**Required (from htr2hpc `pyproject.toml` dev group):**

```
# runtime
pucas>=0.11, psutil, requests, kraken, humanize, simple_slurm, tqdm, intspan, fabric, django~=4.2

# test
pytest>=8.3, pytest-django>=4.9, pytest-cov>=6.1, pytest-mock>=3.14
coverage[toml]>=7.6, responses>=0.25

# code quality
pre-commit
```

**Not required:** full eScriptorium stack (Celery, Redis, PostgreSQL, uWSGI, daphne, channels).

**Django version constraint:** currently `~=4.2` to match eScriptorium. Both v1.0.0 and 26.04 still use Django 4.2, so this constraint is unchanged in either upgrade path.

### 2.4 CI/CD

**Purpose:** run automated tests on GitHub Actions on every PR.

Same as local development/testing, with these differences:

- `pre-commit` handled separately in CI (not installed via dev group)
- GPU-dependent kraken functionality is mocked in tests

CI runs on a single Python version: **3.11** (confirmed in `unit-tests.yml` — no matrix, uses `uv sync` with a committed `uv.lock`). The pyproject.toml classifiers list 3.11 and 3.12, but classifiers are metadata only, not CI configuration. Python 3.11 satisfies all three kraken version constraints (5.3.0 and 6.x require `>=3.9,<3.13`; 7.x requires `>=3.10,<3.14`).

### 2.5 Documentation Build

**Purpose:** build Sphinx docs hosted at readthedocs.io.

**Required (from htr2hpc `pyproject.toml` docs group):**

```
sphinx>=7.2
alabaster
django~=4.2
myst-parser
```

Minimal — no eScriptorium dependencies needed. Django is included because htr2hpc modules import Django models/settings for autodoc.

______________________________________________________________________

## 3. Key Findings

### 3.1 What changes: dev-0.14.2 → v1.0.0

- Kraken constraint: **~=5.2 → ~=6.0** (major version bump in constraint; see below)
- `django-solo` added
- Django 4.2: unchanged

**Upgrade risk: low at the dependency level.** As shown in Section 1.4, numpy 2.x, shapely 2.x, torch ~=2.4, scikit-\*, and coremltools 8.x were all already adopted in kraken 5.3.0 — which is what dev-0.14.2 installs today. The 5→6 constraint change at the dependency level adds only `htrmopo`, `iso639-lang`, and `platformdirs`. htr2hpc has zero direct imports of numpy, shapely, or torch.

Note: the `kraken.containers` `tags` data contract change (Finding 3.5) is a code-level mismatch affecting segmentation training on HPC. The HPC environment was not uniformly running kraken 6.0.3 since initial setup — see Finding 3.5 for the affected user population. It is independent of the web upgrade and should be addressed regardless of which upgrade path is chosen.

### 3.2 What changes: dev-0.14.2 → 26.04

All of the above, plus:

- Kraken constraint: **~=5.2 → ~=7.0.1** (two major versions in constraint)
- `dfine-kraken~=0.4.1` added — D-FINE segmentation model support, pulls in `timm` and `torchmetrics[detection]`
- `pyspark==4.1.1` added as an **explicit version pin** — pyspark is already installed in the current deployment as a transitive dependency of `passim` (confirmed: pyspark 4.1.2 present in staging venv). The 26.04 pin was introduced in upstream commit `f76c1896` (MR !531) because pyspark 4.2.x breaks passim's `seriatim` at startup. This is not a new dependency; it is an upper bound added to a dependency that already exists. See Finding 3.7 for full analysis.

**Two additional API changes in the 5→7 path requiring verification:**

- `build_binary_dataset` (`kraken.lib.arrow_dataset`): `ignore_splits` default changed from `False` to `True` in 7.0.1. htr2hpc does not pass this argument explicitly, so it will silently pick up the new default.
- `serialize` (`kraken.serialization`): gained `sub_line_segmentation=True` as a new parameter in 6.0.0, changing default serialization behavior for documents with sub-line annotations.

**Upgrade risk:** higher than v1.0.0 path due to `dfine-kraken` additions and the two API behavior changes above.

### 3.3 Recommended upgrade path

| Factor | dev-0.14.2 → v1.0.0 | dev-0.14.2 → 26.04 |
|---|---|---|
| Kraken dependency change | minimal (5.3.0 already has numpy 2, shapely 2) | adds safetensors, packaging, htrmopo bump |
| New heavyweight deps | None | dfine-kraken (timm, torchmetrics[detection]) |
| pyspark / JVM | already present (passim transitive dep) | version pin added only; no new cost |
| Django version | 4.2 (unchanged) | 4.2 (unchanged) |
| API behavior changes | tags mismatch (segmentation training only; runtime unverified) | tags fix + ignore_splits + sub_line_segmentation |
| Upgrade risk | Lower | Higher |
| D-FINE segmentation support | No | Yes |

**Important:** "maintain current status" is not a safe baseline. Finding 3.5 identifies a code-level mismatch in the segmentation training path that is inferred (from source inspection) to cause TAGREFS to be silently dropped on HPC for users running kraken 6.0.3 — runtime behaviour has not yet been observed directly. **The `tags` fix is independent of the web-side upgrade** — `train/data.py` runs only on HPC, and the web side does not import `htr2hpc.train.*` (training is triggered remotely via SSH). However, the fix is blocked on §5.4 Step 0 (env audit) — it cannot safely ship while any adroit env is still on kraken 5.x. See Finding 3.5 for details.

What *is* coupled to the web upgrade is **pinning `kraken` in htr2hpc's `pyproject.toml`** (see Note 1). The production deployment uses a single shared virtualenv: eScriptorium's `requirements.txt` installs first, then htr2hpc is layered on top. If htr2hpc pins `kraken~=6.0` while eScriptorium still pins `~=5.2`, the install will conflict. Therefore: pin htr2hpc's `kraken` constraint only after — and in sync with — the web-side upgrade to v1.0.0.

Recommended sequencing:

1. **Run §5.4 Step 0 first** — audit each user's adroit env kraken version
2. If all envs are on 6.0.3: fix `tags` in `train/data.py` immediately (§5.1)
3. If any envs are on 5.x: upgrade them first (§5.4 Step 2), then fix `tags`
4. Upgrade web to v1.0.0 and simultaneously pin htr2hpc `kraken~=6.0`, update `uv.lock`
5. Assess 26.04 separately after v1.0.0 is stable

Since Django 4.2 is unchanged in both paths, no Django migration work is required either way.

### 3.4 HPC and web are on different Kraken major versions — model compatibility requires retroactive audit

Training has already been run on HPC. The question is no longer "should we test this before relying on training runs" — it is "have any models trained on kraken 6.x been successfully loaded and used on the web side (kraken 5.x), and if not, are those models still recoverable?"

The cheapest first step is not to run an experiment: **ask users directly** whether anyone has taken a model produced on HPC and loaded it in eScriptorium's web interface. If someone has done this successfully, the compatibility question is answered. If no one has, or if attempts failed silently, a controlled test is needed.

Controlled test (if user reports are inconclusive): train a minimal model on HPC using kraken 6.0.3, attempt to load it on the staging web venv (kraken 5.3.0), and document whether it succeeds or fails with a specific error. If it fails, any model produced by a user with a kraken 6.x env since 2026-06-02 is potentially unusable on the web side — which would be a data loss finding, not just a compatibility note.

Additionally, the HPC setup installs htr2hpc from `git+https://github.com/Princeton-CDH/htr2hpc.git@develop` — a moving target, not a pinned release. Any breaking change merged to `develop` will affect the next HPC user setup without a versioned rollback path.

### 3.5 `kraken.containers` `tags` data contract changed in 5→6 — code-level mismatch confirmed; runtime behaviour on 6.0.3 inferred but not yet observed

The `tags` field on `BaselineLine` and `Region` changed structure between kraken 5.x and 6.x:

| Version | `tags` structure |
|---|---|
| kraken 5.x | `{"type": "default"}` — flat `str → str` |
| kraken 6.x and 7.x | `{"type": [{"type": "default"}]}` — `str → list[dict]` |

**Scope: segmentation training only.** `train/data.py` has two paths:

- **Recognition training** (when `transcription_id` is present): passes the `Segmentation` object directly to `build_binary_dataset(format_type=None)`. `arrow_dataset.py` does not read `tags` in this path. Recognition training is **not affected** by this finding.
- **Segmentation training** (no `transcription_id`): calls `serialize()` to produce an ALTO file, which is then passed to kraken. This is the affected path.

In the segmentation training path, `train/data.py` constructs `BaselineLine` and `Region` with `tags={"type": line_types.get(line.typology, "default")}` — the kraken 5.x format. On kraken 6.x, `serialization.py:132` iterates over `tags` values expecting `list[dict]`:

```python
types.extend((k, t['type']) for t in v if 'type' in t)
```

When `v` is the string `"handwritten"` (5.x format), `for t in v` iterates over individual characters. `'type' in t` is always False (single chars can't contain the string `'type'`), so `types` stays empty. **The Python generator logic has been verified locally:**

```
v = 'handwritten'  →  list(v) = ['h', 'a', 'n', 'd', ...]
result = []        ←  no character passes 'type' in t

v = [{'type': 'handwritten'}]  →  result = [('type', 'handwritten')]  ✓
```

Serialization on staging (kraken 5.3.0) with the 5.x tags format produces correct TAGREFS:

```xml
<OtherTag DESCRIPTION="line type" ID="LINE_TYPE_1" TYPE="type" LABEL="handwritten"/>
<TextLine ... TAGREFS="LINE_TYPE_1">
```

**What has not been verified:** `serialize()` called on kraken 6.0.3 with 5.x format tags input — this is the directly relevant case. The HPC conda env contains kraken 6.0.3; running a minimal `Segmentation` through `serialize()` there would close the gap. Until then, the outcome is inferred from the source code, not observed.

**Downstream consequence if confirmed:** when TAGREFS are absent from the ALTO file, kraken assigns all lines to the default type during training. A segmentation model trained without TAGREFS cannot distinguish line types, regardless of how the training data was originally annotated.

**Affected scope.** Not all HPC training runs are affected. `user_setup.sh` creates a per-user conda env at `/scratch/network/$USER/.conda` on adroit — each user who runs the setup script gets their own env under their Princeton netid. The script only installs `kraken==6.0.3` when creating a fresh env; it does not upgrade an existing one.

The kraken version installed by `user_setup.sh` changed twice:

- Before 2026-06-02: kraken was installed via conda (kraken 5.x), first on della, then on adroit after the 2025-10-20 migration.
- 2026-06-02 commit `69fb0185`: switched to `pip install` of a locally-cached kraken 6.0.4 wheel. Note: kraken 6.0.4 does not exist on PyPI (6.x releases are 6.0.0, 6.0.2, 6.0.3 only); this wheel was a local or non-public build. Any env created on this exact date carries a kraken version of unknown provenance. The window is one day and the risk is likely zero, but should be noted if `pip show kraken` returns 6.0.4 during Step 0 of §5.4.
- 2026-06-03 commit `65ac3ca5`: changed to `pip install kraken==6.0.3` (current).

The affected scope is therefore: **segmentation training runs performed by users who created a fresh adroit env on or after 2026-06-02.** Anyone who built their env before that date is on kraken 5.x and is not affected. The `--reinstall-htr2hpc` flag only reinstalls htr2hpc, not kraken — so older envs remain on kraken 5.x.

To confirm which envs fall in scope: each user who has used HPC training should run `conda activate htr2hpc && pip show kraken` on adroit and report the version. This cannot be checked remotely (the web VM `pulsys` account does not have adroit access).

The same assumption exists in `lib/dataset/utils.py::_get_type` (consistent across 6.0.3 and 7.0.3).

The fix is to update `tags` construction in `train/data.py` to the 6.x format:

```python
{"type": [{"type": line_types.get(line.typology, "default")}]}
```

**Important: this fix is not safe to ship while any user's adroit env is still on kraken 5.x.** On kraken 5.3.0, `serialization.py:128` collects `page['line_types']` by directly using the `tags` values as labels. Feeding it the 6.x format `[{'type': 'handwritten'}]` instead of the string `'handwritten'` will not crash, but the ALTO `LABEL` attribute will be output as a stringified dict rather than a plain type name — the output is still wrong, just differently wrong:

| Env kraken | Current `tags` format | After fix |
|---|---|---|
| 5.x | ✅ correct TAGREFS | ❌ label becomes `[{'type': 'handwritten'}]` |
| 6.0.3 | ❌ TAGREFS silently absent | ✅ correct TAGREFS |

The fix can only be correct for one major version at a time. There is no plan to support multiple kraken versions simultaneously — the goal is to set a minimum version and ensure the code works correctly with it. The required sequencing is:

- **Preferred:** run Step 0 of §5.4 first to confirm no user is still on kraken 5.x. If all envs are on 6.0.3, ship the fix immediately.
- **If some envs are on 5.x:** upgrade them first (add `--upgrade-kraken` to `user_setup.sh`, §5.4 Step 2), then ship the fix. Do not attempt to support both formats simultaneously.

**This means §5.4 Step 0 blocks §5.1. See §5.4 for the env audit.**

This fix is independent of the web-side upgrade (see Section 3.3).

### 3.6 Django is a de facto HPC dependency via `api_client.py` — slugify output must match eScriptorium

`src/htr2hpc/api_client.py:13` imports `from django.utils.text import slugify` (the original author left the comment `# is django a reasonable dependency?`). Because `train/run.py` imports `api_client`, Django is pulled into the HPC training execution path at runtime.

The slugify call is at `api_client.py:488`:

```python
slugify(document_name).replace("-", "_")[:32]
```

This generates part of the export filename: `export_doc{document_id}_{slug}_{format}_{timestamp}.zip`. The **same logic** exists in eScriptorium at `app/apps/imports/export.py:60`:

```python
slugify(self.document.name).replace("-", "_")[:32]
```

The comment in `api_client.py:482–485` explicitly states the logic was "copied & adapted from escriptorium import.export.BaseExporter logic." The two codebases generate the same slug independently — htr2hpc to construct a URL to fetch the export, eScriptorium to generate the filename it serves. **They must produce identical output on the same input, or the download will fail with a 404.**

This is why Rebecca's question matters: replacing `django.utils.text.slugify` with `python-slugify` is not safe without verifying the outputs match on the exact inputs used in production (document names containing non-ASCII characters, punctuation, and Unicode). Django's `slugify` and `python-slugify` differ in their Unicode normalisation and transliteration behaviour — for ASCII-only document names the output is identical, but for names with accented characters or non-Latin scripts the results can diverge.

**Options:**

1. **Keep `django.utils.text.slugify`** — no correctness risk, but Django remains an HPC dependency. This is the safest choice if the `[django]` extra split (§5.3) is adopted, because Django will still be installed on HPC via htr2hpc's core deps or the explicit `htr2hpc[hpc]` extra would need to carry it.

2. **Replace with `python-slugify`** — reduces HPC footprint, but requires testing that outputs match for all document names in the Princeton corpus. A mismatch would silently cause export downloads to fail for documents with non-ASCII names. If the team can confirm the Princeton document corpus contains only ASCII names, this is safe. Otherwise: **unverified, requires testing before adopting**.

3. **Vendor a one-line equivalent** — Django's `slugify` implementation is ~10 lines using `unicodedata.normalize` + `re.sub`. A local copy avoids the Django dependency and the `python-slugify` divergence risk, at the cost of duplicating logic that eScriptorium already maintains. This is only appropriate if the team is confident the document name character set is stable and well-understood.

### 3.7 passim / pyspark — feature confirmed unused; removal recommended; future re-enablement documented

#### Status (confirmed 2026-08-10)

**Christine Roughan (instance administrator) confirmed:** the Align feature is not currently in use at Princeton and has not been enabled. It is on the list as a future "Nice to Have." **Rebecca Koeser confirmed** interest in re-enabling it in the future, with a note that passim has changed significantly across versions.

This closes the open question from the previous version of this document. The remediation path is now clear: **remove passim for now (Option 1), with a documented path to re-enable**.

#### Current state

`passim` is installed in all three eScriptorium versions (dev-0.14.2, v1.0.0, 26.04) via the same git dependency:

```
git+https://github.com/dasmiq/passim.git@v2.0.1
```

passim declares `pyspark>=3.0.1`, which causes pyspark — and the JVM it requires — to be installed as a transitive dependency. Confirmed on staging: pyspark 4.1.2 and passim 2.0.0 are present in `/srv/www/escriptorium/app/env/`.

The feature is gated by `TEXT_ALIGNMENT_ENABLED` (`app/escriptorium/settings.py:453`), which defaults to `False`. The alignment task is routed to the `jvm` Celery queue (`app/escriptorium/settings.py:241`), but no Princeton ansible config defines a worker consuming that queue — so alignment jobs would queue indefinitely and never execute even if the setting were turned on.

passim is web-side only. It has never been part of the HPC environment.

#### Passim version history since v2.0.1

passim has no formal release after v2.0.1 (February 7, 2024), but the `main` branch has received 21 commits through March 27, 2026. Key changes:

- **Sept 26, 2025:** major upgrade to **pyspark 4.0.1** and graphframes 0.9.3. This is a breaking change from pyspark 3.x. eScriptorium's 26.04 tag adds `pyspark==4.1.1` as an explicit pin because pyspark 4.2.x breaks passim's `seriatim` startup (upstream commit `f76c1896`, MR !531).
- **Jan–March 2026:** sharding implementation for pairs, dfpost, and extents — performance improvement for large corpora.
- **March 27, 2026:** fix to `passLocs` for pairwise alignment.
- **May 2025:** new `--to-index`, `--to-pairs`, `--to-extents` checkpoint flags and `--refpref` option.

**Impact for future re-enablement:** v2.0.1 requires pyspark 3.x; the current `main` requires pyspark 4.0.1+. If Princeton re-enables Align, the version of passim to install will need to match the available pyspark version. The eScriptorium 26.04 pin (`pyspark==4.1.1`) aligns with the Sept 2025 passim `main` branch, not with v2.0.1. There is no tagged release of passim that ships the Spark 4.x upgrade — it would need to be installed from a specific commit.

#### Recommended action: remove passim now

Remove `passim` from `requirements.txt`. pyspark disappears as a transitive dependency. No functional impact — the feature has never been active.

#### To re-enable Align in the future

The following steps are required to restore the feature:

1. **Add back `passim`** to `requirements.txt` — pinned to a specific commit on `main` that includes the Spark 4.x upgrade (or to a future tag if one is published). The v2.0.1 tag is no longer suitable if pyspark 4.x is the target.
2. **Install a JDK** on the web VMs — add a JDK install task to the ansible role. `java` is not currently on `PATH` on the staging server.
3. **Configure a `celery-jvm` worker** in ansible's `supervisor_programs` — add an entry consuming the `jvm` queue on at least one web VM. Without this worker, alignment jobs queue indefinitely and never execute.
4. **Set `TEXT_ALIGNMENT=True`** in the Princeton environment config — the feature is gated by this environment variable, which defaults to `False`.

#### Cost breakdown (for reference if feature is re-enabled)

| Cost | Where it lands | Mitigation if re-enabling |
|---|---|---|
| Disk space | Web VMs | Accept, or use isolated venv (Option 3 from prior draft) |
| JVM memory at runtime | Web VMs, contends with uWSGI / Celery / PostgreSQL | Restrict `celery-jvm` worker to one VM via Celery queue routing |
| Version coupling | Shared venv dependency resolver | Isolated venv removes pyspark from main venv resolution |

______________________________________________________________________

## 4. Notes

### Note 1: htr2hpc's `kraken` dependency and lockfile drift

htr2hpc declares `kraken` with no version constraint in `pyproject.toml`. In production this is fine because the eScriptorium `requirements.txt` installs a pinned Kraken version first, and htr2hpc's unpinned dep is satisfied by whatever is already installed (see caveat in Section 2.2 about pip ordering).

In CI, `uv sync` reads the committed `uv.lock`, which currently locks kraken at **7.0.2** — two major versions ahead of production (kraken ~=5.3.0). Note that `uv.lock` governs only CI and local development; production uses ansible + pip directly and does not consume the lockfile. The risk is that the lockfile has drifted from the production version with no declared target, and there is no mechanism to keep them in sync. If a htr2hpc code path works with kraken 7.0.2 but fails with kraken 5.3.0, the test suite will not catch it.

Once a target eScriptorium version is confirmed, the `kraken` dependency in htr2hpc's `pyproject.toml` should be pinned to match — e.g. `kraken>=6.0,~=6.0` for a v1.0.0 target — and `uv.lock` regenerated accordingly. This pin must be done in sync with the web-side upgrade (see Section 3.3).

### Note 2: Local eScriptorium repository had incorrect upstream remote

During the preparation of this review, the local eScriptorium clone was found to have its `origin` remote pointing to `UB-Mannheim/escriptorium` (a fork) rather than `gitlab.com/scripta/escriptorium` (the canonical upstream). The remote has since been corrected and the local `develop` branch reset to upstream.

Three questions this raises:

1. **What code has been running in production?** The Princeton deployment is managed via ansible (`escriptorium_setup`) and deploys from a GitLab tag — not from the local clone. Production is unaffected by the local remote misconfiguration. However, this should be confirmed explicitly: verify that the ansible `ref` variable in the last deploy matches a tag on `gitlab.com/scripta/escriptorium`, not on the UB-Mannheim fork.

2. **Are the ESC-2026 security patches still in place?** Per Note 3, those patches are applied at the ansible layer, not in the eScriptorium repository itself — so the remote reset should not have affected them. This should be verified before the next deploy.

3. **Are the three columns in Section 1.1 trustworthy?** The dependency table was built by fetching `requirements.txt` from each tag via the GitLab API (`gitlab.com/scripta/escriptorium`), not from the local clone. The data is independent of the local remote. Three rows were spot-checked line by line against the raw API response; no discrepancies were found.

### Note 3: ESC-2026 security patches and upstream merge status

The ansible deployment (`escriptorium_setup/tasks/main.yml`) applies ESC-2026 security patches as a temporary workaround, with a comment noting they should be removed once Princeton upgrades past the patched version. Before deploying any eScriptorium upgrade (v1.0.0 or 26.04), the team should verify whether the ESC-2026 fixes have been merged into the upstream eScriptorium release being targeted. If they have not been merged, the security patches will need to be re-applied to the new version (or re-submitted upstream) to avoid reintroducing the vulnerabilities.

### Note 4: `passim` / `pyspark` git dependency — web deployment only

All three eScriptorium versions install `passim` from a git commit (`git+https://github.com/dasmiq/passim.git@v2.0.1`), which bypasses PyPI and fetches directly from GitHub at install time. The concern about outbound GitHub access at deploy time applies to the **web application deployment** only — passim is not installed on HPC (see Section 2.1). For full analysis of pyspark cost, the Align feature status, and remediation options, see Finding 3.7.

______________________________________________________________________

## 5. Dependency Management Proposals

This section collects the actionable items from this review into concrete proposals for team decision and issue creation. The issue text references the relevant finding for context.

### 5.1 Fix `tags` construction in `train/data.py` — blocked on §5.4 Step 0

**Priority: high, but blocked on env audit (§5.4 Step 0) — cannot safely ship until all adroit envs are confirmed on kraken 6.0.3.**

The `tags` format passed to `BaselineLine` and `Region` uses the kraken 5.x structure. Source inspection of `serialization.py:132` and local simulation of the generator logic both indicate TAGREFS would be silently absent in the ALTO output on kraken 6.0.3. Affects segmentation training only (see Finding 3.5 for scope, sequencing, and forward-compatibility constraint).

**Proposed action:**

1. **First: run §5.4 Step 0** — audit each user's adroit env to confirm all are on kraken 6.0.3 (or upgrade them if not). This unblocks the fix.

2. Verify the fix on HPC: construct a minimal `Segmentation` object (two `BaselineLine` + one `Region`, 5.x tags format) and call `kraken.serialization.serialize()` in the kraken 6.0.3 conda env. Check ALTO output for TAGREFS. Requires no torch and no real document.

3. Update `tags` construction in `train/data.py` to the 6.x format and open a PR.

4. Add a test to cover this. Rebecca asked for a unit or integration test ("more likely integration test") that confirms the tags structure is correct for the installed kraken version.

   **Test feasibility assessment:**

   The functions that exercise this code path — `get_segmentation_data()`, `serialize_segmentation()`, `get_training_data()` at `train/data.py:100–160` — are currently **untested** (confirmed: no test in the suite calls them). They accept Django ORM objects (`DocumentPart`, `Line`, `Block`) as input.

   - **Unit test:** would require either (a) constructing real Django ORM objects via the test database (feasible with `pytest-django` and SQLite; no eScriptorium stack needed) or (b) mocking them with `MagicMock`. The key assertion is on the `serialize()` output — specifically that `TAGREFS` appear in the ALTO XML. This *can* be done in unit test scope because `kraken.serialization.serialize()` does not require torch or GPU: it takes a `Segmentation` object (pure Python dataclass) and produces an XML string. Confirmed: `serialize()` has no torch import in the call path for ALTO output.
   - **Integration test:** would call the full `serialize_segmentation()` stack against a real DocumentPart. This exercises more code but requires a populated test database, which adds setup complexity. Rebecca's "more likely integration test" judgement was likely based on the assumption that mocking kraken would be needed — but since `serialize()` is pure Python and kraken is already installed in CI (`uv.lock` pins 7.0.2), a unit test using a real `Segmentation` object is viable without mocking.

   **Recommendation:** write a **unit test** at `tests/test_train_data.py` that:

   - Constructs a `Segmentation` directly (two `BaselineLine`, one `Region`, 5.x tags format)
   - Calls `kraken.serialization.serialize(segmentation, image_size=(100, 100))`
   - Parses the ALTO XML output and asserts that `TAGREFS` attributes are non-empty on `<TextLine>` elements

   This is runnable in CI without torch and without Django ORM fixtures. It also directly documents the data contract. If the team wants broader coverage (full `get_segmentation_data()` → `serialize_segmentation()` stack), that can be an integration test added separately.

**This finding warrants opening a GitHub issue to track the fix.**

**Suggested issue title:** `Fix kraken 6.x tags format in train/data.py`

______________________________________________________________________

### 5.2 Pin `kraken` in `pyproject.toml` — in sync with web upgrade

**Priority: medium — do not do this before the web upgrade decision is made.**

htr2hpc's `kraken` dependency is currently unpinned. In production the eScriptorium `requirements.txt` controls which version is installed, so there is no conflict today. But CI's `uv.lock` is locked at 7.0.2, two major versions ahead of production, meaning the test suite does not cover the version actually running in production.

**Proposed action:** once the target eScriptorium version is confirmed, pin htr2hpc's `kraken` to match and regenerate `uv.lock`:

| Target eScriptorium | Proposed htr2hpc `kraken` constraint |
|---|---|
| v1.0.0 | `kraken>=6.0,~=6.0` |
| 26.04 | `kraken>=7.0,~=7.0` |

This must be done in the same PR as the web upgrade, because the shared production virtualenv installs both requirements files and a version mismatch will cause a conflict. Note that `uv.lock` governs only CI and local development — updating it does not affect production.

**Suggested issue title:** `Pin kraken version in pyproject.toml to match eScriptorium target`

______________________________________________________________________

### 5.3 Split htr2hpc dependencies by environment: `[django]` and `[hpc]` optional extras

**Priority: medium — reduces install footprint on both sides, but not urgent.**

**Rebecca's direction:** call the web-side optional group `[django]` (not `[webapp]`). This also requires a CI test matrix covering with-Django and without-Django cases (see below).

#### Complete package attribution table

Every runtime dependency from `pyproject.toml [project.dependencies]`, one row per package. All import locations verified by source search.

| Package | Group | Imported at (file:line) | Rationale |
|---|---|---|---|
| `kraken` | `[hpc]` | `train/data.py:8–9,13`; `train/run.py:15` | Used only in `train/`. `tasks.py` does **not** import kraken directly — it imports `ensure_kraken_version` from `htr2hpc.train.hpc` (`tasks.py:17`), but that is a htr2hpc-internal import, not a direct kraken import. No web-side code calls kraken at runtime. |
| `requests` | core | `api_client.py:12`; `train/run.py:13` | `api_client` is imported by both `tasks.py` (web) and `train/run.py` (HPC). `train/run.py:97` catches `requests.exceptions.ConnectionError` directly. Both sides make HTTP calls to the eScriptorium API. |
| `humanize` | core | `api_client.py:11` | `api_client` is shared (see `requests` above); `humanize.naturalsize()` at `api_client.py:545,576` is called during file downloads, which happen on both the web side and the HPC side. |
| `intspan` | core ⚠️ | `tasks.py:13`; `train/run.py:14` | **Cross-side string contract** — see section below. `tasks.py:226–227` serialises part IDs as an intspan string; `train/run.py:71` receives and parses that same string. The comment at `tasks.py:226` is explicit: *"parse and serialize with intspan since that's what we use on the other side."* Cannot be placed in either extra without breaking the other side. |
| `simple_slurm` | `[hpc]` | `train/slurm.py:7` | Slurm job submission. Only in `train/slurm.py`; never imported in `tasks.py` or any non-`train/` module. |
| `tqdm` | `[hpc]` | `train/run.py:16`; `train/data.py:14`; `train/rm_models.py:23` | Progress bars. All three import sites are in `train/`. Not used in `tasks.py` or any web-side module. |
| `fabric` | `[django]` | `tasks.py:12` | SSH from the web server *to* HPC. Imported only in `tasks.py` (Celery tasks). The HPC environment does not run fabric — it is the *target* of the SSH connection, not the initiator. |
| `django` | `[django]` | `tasks.py:7–11`; `views.py:1–4`; `urls.py:1`; `context_processors.py:5–6`; `apps.py:1`; `admin.py:1–2`; `users.py:4–5`; `management/commands/cleanup_exports.py:5–6`; `migrations/0001_*.py:3`; `api_client.py:13` | Pervasive Django framework dependency — all web-side modules. The single non-web use is `api_client.py:13` (`django.utils.text.slugify`), which is a cross-side string contract (see below). |
| `pucas` | `[django]` | `admin.py:4` | CAS authentication admin integration. Web only; no usage in `train/`. |
| `psutil` | `[django]` | `context_processors.py:4` | Web-only: Django context processor for VM memory display (`context_processors.py:25`). Not imported anywhere in `train/` or `api_client.py`. |

**Correction from previous draft:** `kraken` was previously listed as `core` under the assumption that `tasks.py` uses kraken directly. It does not — confirmed by source search. `kraken` belongs in `[hpc]`.

**Correction from previous draft:** `psutil` was previously listed as `core` ("cheap, no conflicting deps"). Source confirms it is web-only (`context_processors.py` only). Moved to `[django]`.

______________________________________________________________________

#### Cross-side string contracts

Two packages in the table above impose a constraint stronger than "both sides import it": both sides independently generate a string that must be **byte-for-byte identical** for the system to function. These cannot be split into separate extras or replaced independently on one side without risking a silent mismatch.

**1. `intspan` — part ID serialization**

`tasks.py:226–227` (web/Celery):

```python
# parse and serialize with intspan since that's what we use on the other side
arg_options.append(f"--parts {intspan(part_pks)}")
```

`train/run.py:71` (HPC, `TrainingManager` dataclass):

```python
parts: Optional[intspan] = None
```

The web side serialises a set of document part PKs into an intspan string (e.g. `"1-5,7,9-12"`) and passes it as a CLI argument to the HPC training process. The HPC side parses the same string back into an `intspan` object. If the two sides use different libraries or different versions with different serialisation behaviour, the argument will either be unparseable or silently produce a different set of IDs. This is why the comment names intspan explicitly — it is a documented cross-side contract, not incidental sharing.

**intspan must stay in `core`.**

**2. `django.utils.text.slugify` — export filename matching**

`api_client.py:488` (HPC, constructs download URL):

```python
slugify(document_name).replace("-", "_")[:32]
```

`app/apps/imports/export.py:60` (eScriptorium web, generates filename):

```python
slugify(self.document.name).replace("-", "_")[:32]
```

The comment at `api_client.py:482–485` states this logic was "copied & adapted from escriptorium import.export.BaseExporter logic." The web side generates the export archive filename; the HPC side constructs the URL to download it. Both must produce identical output on the same document name, or the download fails with a 404.

This is the constraint that makes replacing `django.utils.text.slugify` with `python-slugify` risky (see Finding 3.6): Django and python-slugify differ on non-ASCII input. Until the slugify dependency is resolved, `django` cannot be cleanly removed from the HPC execution path — it must either stay in `core` or the `[hpc]` extra must include `django`.

**Consequence for the `[hpc]` split:** the proposed structure below reflects the state *after* the slugify issue (Finding 3.6) is resolved. If shipped before resolution, `django` must be added to `core` or explicitly to `[hpc]`.

______________________________________________________________________

#### Proposed `pyproject.toml` structure

*Prerequisite: Finding 3.6 (slugify) resolved before finalising.*

```toml
[project]
dependencies = [
  # core: required in both web and HPC environments
  "requests",       # api_client.py:12 (shared); train/run.py:13
  "humanize",       # api_client.py:11 (shared)
  "intspan",        # tasks.py:13 + train/run.py:14 — cross-side string contract
]

[project.optional-dependencies]
django = [
  "django~=4.2",    # tasks.py, views.py, urls.py, context_processors.py, etc.
  "pucas>=0.11",    # admin.py:4 — CAS auth
  "fabric",         # tasks.py:12 — SSH from web server to HPC
  "psutil",         # context_processors.py:4 — web VM memory display
]
hpc = [
  "kraken",         # train/data.py:8-9,13; train/run.py:15
  "simple_slurm",   # train/slurm.py:7
  "tqdm",           # train/run.py:16; train/data.py:14; train/rm_models.py:23
]
```

#### Deployment commands

**Ansible (web server)** — install `htr2hpc[django]`:

```yaml
# group_vars/htr/vars.yml or ansible task
python_extra_packages:
  - "git+https://github.com/Princeton-CDH/htr2hpc.git@0.x.y#egg=htr2hpc[django]"
```

**HPC `user_setup.sh`** — install `htr2hpc[hpc]`:

```bash
pip install "git+https://github.com/Princeton-CDH/htr2hpc.git@0.x.y#egg=htr2hpc[hpc]"
```

Note: `htr2hpc[hpc]` does **not** include `django` or `pucas`. The slugify prerequisite (Finding 3.6) must be resolved before deploying this split; otherwise add `django~=4.2` to `[hpc]` temporarily.

#### CI impact — test matrix required

Splitting `django` into an optional extra means the test suite must pass in two configurations:

1. With `[django]` installed — all existing tests run
2. Without `[django]` installed — Django-dependent tests must be skipped or conditionally excluded

The best CDH precedent is **parasolr** (`Princeton-CDH/parasolr/.github/workflows/unit_tests.yml`), which tests against `django=0` (no Django) plus Django 3.2–5.0 in a matrix:

```yaml
- if: ${{ matrix.django }} != 0
  run: pip install Django==${{ matrix.django }} pytest-django
```

For htr2hpc the matrix is simpler — only one Django version (4.2) — so it is just `[with-django, without-django]`.

Tests known to require Django: `test_context_processors.py`, `test_users.py`, parts of `test_api_client.py` (slugify path), `test_cleanup_exports.py` (management command), and `test_tasks.py`. Tests in `test_train_data.py` do not import Django directly and should pass in the no-Django leg.

**Suggested issue title:** `Split django/pucas/fabric/psutil into [django] extra; move kraken to [hpc]; add CI test matrix`

______________________________________________________________________

### 5.4 Pin htr2hpc version on HPC — stop installing from `@develop`; audit per-user env kraken versions

**Priority: medium — currently any `develop` merge silently changes the HPC environment for new setups. Additionally, per-user envs may be running different kraken major versions with no upgrade path.**

`user_setup.sh` creates a per-user conda env at `/scratch/network/$USER/.conda` on adroit. Each user who runs the script gets their own env. The script only installs `kraken==6.0.3` when creating a fresh env — it does not upgrade an existing one. This means:

- A user who ran setup before 2026-06-02 may still have kraken 5.x in their env.
- `--reinstall-htr2hpc` only reinstalls htr2hpc, not kraken.
- There is no mechanism to bring an existing env up to the current expected kraken version.

**Proposed action:**

Step 0 (**blocks §5.1** — do this first): each user with an adroit env should run:

```bash
conda activate htr2hpc && pip show kraken
```

Document the version for each env. If any env is on kraken 5.x, the `tags` fix in §5.1 cannot safely ship until those envs are upgraded (see Step 2 below). If any env reports kraken 6.0.4, note that this is a non-PyPI build (see Finding 3.5); investigate its provenance.

Step 1 (pin to tag): change `@develop` to a specific release tag in `user_setup.sh`. Note: htr2hpc is **not published on PyPI** (`pip install htr2hpc==0.x.y` will fail). Use the git tag form:

**Option A — Pin to a git tag (small change, can be done now):**

```
pip install git+https://github.com/Princeton-CDH/htr2hpc.git@0.7.0
```

**Option B — Publish to PyPI (separate decision, larger scope):**
If the team wants `pip install htr2hpc==0.7.0` to work, htr2hpc needs to be built and uploaded to PyPI. This is a separate process decision that should be tracked independently.

Step 2 (optional): consider adding a `--upgrade-kraken` flag to `user_setup.sh` that reinstalls kraken to the current expected version, so existing envs can be brought in sync without a full teardown.

**Recommended immediate action:** implement Option A now; treat Option B as a separate item if PyPI publication is desired.

**Suggested issue titles:**

- `Audit per-user HPC conda env kraken versions` (Step 0 — immediate)
- `Pin htr2hpc to a release tag in HPC user_setup.sh` (Option A — immediate)
- `Publish htr2hpc to PyPI` (Option B — separate, if desired)

______________________________________________________________________

### 5.5 Verify ansible deploy `ref` against canonical upstream (one-time check)

**Priority: low — informational, but closes an open question from Note 2.**

The local eScriptorium clone had its `origin` remote pointing to the UB-Mannheim fork. Production is deployed via ansible from a GitLab tag, not from the local clone — so production is likely unaffected. But this has not been explicitly verified.

**Proposed action:** check the ansible inventory / group_vars for the last deployed `ref` value and confirm it matches a tag on `gitlab.com/scripta/escriptorium`. Document the result.

**Suggested issue title:** `Confirm production eScriptorium deploy ref points to scripta upstream`

______________________________________________________________________

### 5.6 Audit kraken 5.x / 6.x model compatibility (retroactive)

**Priority: high — training has already been run on HPC. This is a retroactive audit, not a pre-flight check.**

The cheapest first step is not an experiment: **ask users directly** whether anyone has taken a model produced on HPC and loaded it in eScriptorium's web interface. A single confirmed success answers the question. A confirmed failure — or no one having tried — requires a controlled test.

**Proposed action:**

1. Ask HPC users whether anyone has successfully loaded an HPC-trained model in the eScriptorium web UI (kraken 5.3.0). Document the outcome.
2. If no conclusive answer from step 1: train a minimal model on HPC using kraken 6.0.3, attempt to load it on the staging web venv (kraken 5.3.0), and record whether it succeeds or fails with a specific error.
3. If incompatible: any model produced by a user with a kraken 6.x env since 2026-06-02 is potentially unusable on the web side — enumerate the affected models and notify users.

**This finding warrants opening a GitHub issue to track the audit.**

**Suggested issue title:** `Audit kraken 5.x/6.x model binary compatibility — retroactive review of HPC-trained models`

______________________________________________________________________

### 5.7 Verify ESC-2026 security patches against upgrade target (hard prerequisite for any web upgrade)

**Priority: high — must be confirmed before deploying v1.0.0 or 26.04.**

The ansible deployment applies ESC-2026 security patches as a temporary workaround. Before deploying any eScriptorium upgrade, the team must confirm whether these fixes have been merged into the target upstream release. If they have not, the patches must be re-applied to the new version or submitted upstream to avoid reintroducing the vulnerabilities.

**Proposed action:**

1. Identify the specific commits / MRs that comprise the ESC-2026 patches (from `escriptorium_setup/tasks/main.yml`).
2. Check whether each is present in the target upstream tag (v1.0.0 or 26.04) via GitLab MR history or `git log`.
3. If missing: either re-apply the patches as a Princeton-local branch on top of the target tag, or open MRs upstream.

**Suggested issue title:** `Confirm ESC-2026 security patches are present in eScriptorium upgrade target`

______________________________________________________________________

### 5.8 Remove passim from `requirements.txt`; document re-enablement path

**Priority: medium — feature confirmed unused (Christine Roughan, 2026-08-10). Removal is straightforward; no upgrade decision required.**

passim has never been enabled at Princeton. The open question from the previous version of this document is closed. See Finding 3.7 for full confirmation details, passim version history, and the four steps required to re-enable in the future.

**Proposed action:**

1. Remove `passim` from eScriptorium's `requirements.txt` in the Princeton deployment (apply as a local patch at the ansible layer, or submit upstream if the upstream team agrees).
2. Verify the venv rebuilds cleanly without passim and pyspark.
3. Keep Finding 3.7's re-enablement checklist in this document so the steps are not lost.

This can be done independently of the upgrade decision — passim is pinned to the same git ref in all three eScriptorium versions.

**Suggested issue title:** `Remove passim from requirements.txt — feature unused; document re-enablement path`
