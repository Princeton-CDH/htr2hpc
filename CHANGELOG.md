# CHANGELOG

<!-- DRAFT — generated from git log + closed issues/PRs; review before merge -->

## 0.8

- Update to eScriptorium v1.0.1; constrain kraken to 6.x to match
- Auto-check and upgrade Kraken version on HPC before training begins
- Fix kraken 6.x val_accuracy parsing for recognition and segmentation tasks, including wide-column SLURM output
- Fix kraken 6.x tags format for BaselineLine and Region in ALTO export
- Propagate error and canceled status to all secondary task reports
- Pass anaconda module as CLI flag instead of importing from Django settings
- Make `HTR2HPC_GITREF` setting optional; fall back to `__version__` when not set
- Add file retention notice to export email notification

<!-- END DRAFT -->

## 0.7

- New accounts created via CAS login are inactive by default
- Integrate Django admin functionality for initializing CAS users (via django-pucas v0.11)
- Add `cleanup_exports` management command to remove old user export files
- SSH key label in profile setup instructions now reflects the configured Django site domain
- `createcasuser` now accepts multiple NetIDs in a single command
- Set up Sphinx documentation; configure to publish on ReadTheDocs
- Add devbox support for simplifying local development setup

## 0.6

- Switch eScriptorium to use Adroit HPC cluster, including updated scratch paths and monitoring links
- Update homepage heading text for production instance
- Display htr2hpc version in site footer with link to GitHub repo
- Add pre-commit hooks for code quality (ruff, codespell, yamlfmt, mdformat, uv, action-validator)
- Add `DEVELOPERNOTES.md` with instructions for development setup, creating a release, and deploying with Ansible
- Update to kraken 6.0.3; kraken 6 dropped conda support so switch HPC setup to use pip install instead of conda

## 0.5

- Initial release of htr2hpc for beta testing.
