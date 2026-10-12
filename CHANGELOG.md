# CHANGELOG

## 0.8

- Update to eScriptorium v1.0.1, with the following changes to match the new version:
  - Constrain kraken to 6.x
  - Fix accuracy reporting for recognition and segmentation training jobs
  - Update ALTO conversion to generate BaselineLine and Region tags in kraken 6.x format
  - Task errors and cancellations are reflected in all associated task reports (now 1 per page in eScr 1.0)
- Training jobs on HPC automatically upgrade `htr2hpc` to the version matching the deploy, and updates all dependencies (particularly Kraken)
- Configure a cron job to automatically clean up old user export files, and add file retention notice to export email notifications

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
