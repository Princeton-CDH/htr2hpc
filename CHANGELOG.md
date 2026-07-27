# CHANGELOG

## 0.7

- New accounts created via CAS login are inactive by default
- Add Django admin interface for CAS user initialization
- Add `cleanup_exports` management command to remove old user export files
- SSH key label in profile setup instructions now reflects the configured site domain

## 0.6

- Switch eScriptorium to use Adroit HPC cluster, including updated scratch paths and monitoring links
- Update homepage heading text for production instance
- Display htr2hpc version in site footer with link to GitHub repo
- Add pre-commit hooks for code quality (ruff, codespell, yamlfmt, mdformat, uv, action-validator)
- Add `DEVELOPERNOTES.md` with instructions for development setup, creating a release, and deploying with Ansible
- Update to kraken 6.0.3; kraken 6 dropped conda support so switch HPC setup to use pip install instead of conda



## 0.5

- Initial release of htr2hpc for beta testing.
