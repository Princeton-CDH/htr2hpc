# CHANGELOG

## 0.6

- Switch HPC cluster from Della to Adroit, including updated scratch paths (`/scratch/network/`) and monitoring links
- Extend model list template to include sources for public models
- Add lock file to `user_setup.sh` to prevent multiple simultaneous instances
- Relocate `.conda` to scratch on Adroit to handle limited home directory space
- Pin `rich<14.1.0` to fix compatibility issue with newer versions
- Fix duplicate notification messages during setup
- Fix typos in `api_client.py` docstrings
- Remove custom `contactus` and model list HTML overrides
- Add `DEPLOY_NOTES.md` with release and deployment instructions
- Update homepage heading text for production instance
- Display htr2hpc version in site footer with link to GitHub repo
- Add pre-commit hooks for code quality (ruff, codespell, yamlfmt, mdformat, uv, action-validator)

## 0.5

- Initial release of htr2hpc for beta testing.
