# CHANGELOG

## 0.6

- Switch eScriptorium to use Adroit HPC cluster, including updated scratch paths and monitoring links
- Fix Ansible playbook deploy ordering so nginx restarts after the escriptorium_setup patch is applied
- Fix Celery worker concurrency configuration
- Update homepage heading text for production instance
- Display htr2hpc version in site footer with link to GitHub repo
- Add pre-commit hooks for code quality (ruff, codespell, yamlfmt, mdformat, uv, action-validator)



## 0.5

- Initial release of htr2hpc for beta testing.
