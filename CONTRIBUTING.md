# Contributing to htr2hpc

This guide covers the day-to-day workflow for CDH team members contributing to htr2hpc — from opening a pull request through deploying to staging for user testing.

For local development setup, running tests, and building documentation, see [DEVELOPERNOTES.md](DEVELOPERNOTES.md).

______________________________________________________________________

## Manual checks before requesting review

Run the test suite locally before marking the PR ready for review:

```sh
uv run pytest
```

Coverage targets (enforced by Codecov): 60% overall, 100% on test files themselves.
If your change touches the HPC training path or the UI, smoke-test it manually against a local or staging instance before requesting review.

Pre-commit hooks (ruff, mdformat, codespell, etc.) run automatically on `git commit` once you've run `pre-commit install`. If you haven't set that up yet, see [DEVELOPERNOTES.md](DEVELOPERNOTES.md).

______________________________________________________________________

## CodeRabbit review

CodeRabbit posts an automated review on every PR. We use the web dashboard configuration (no `.coderabbit.yaml` in the repo). A few things to know about our setup:

- **High-level summary is disabled.** We turned it off because it was prone to hallucination and not reliable enough to be useful.
- **Suggested labels are disabled.** We don't consistently label PRs right now. If we decide to start using labels, consider re-enabling this along with "Auto apply labels" under Statuses.
- **Docstrings finishing touch is disabled.** If you want AI-assisted docstring generation, check the tool Laura shared in Slack instead.

Guidelines for handling CodeRabbit feedback:

- **Address blocking suggestions** before merging, especially around test coverage and error handling.
- **Nitpick / style suggestions** are optional. If you decide not to act on one, reply to the comment with a brief explanation so reviewers know it was considered.
- Unit test suggestions from CodeRabbit are generally worth acting on — they tend to catch real gaps.

______________________________________________________________________

## Human code review

At least one human approval is required before merging. Reviewer checklist:

- [ ] Pre-commit hooks pass (no linting or formatting errors)
- [ ] CI unit tests pass
- [ ] CodeRabbit comments addressed or replied to
- [ ] New or changed behavior is covered by tests

Note: keep the code review checklist and the acceptance testing checklist separate. Some tests cannot be verified through code review alone — for example, anything that requires running the full application against real HPC infrastructure or live data. Those items belong in acceptance testing, which Christine runs on staging after the feature is deployed. Rebecca's code review covers what can be assessed from the code itself; it should not be blocked on tests that require Christine to run the application manually.

______________________________________________________________________

## Deploying to staging for user testing

Once a feature is merged to `develop`, deploy it to the staging server (`test-htr.princeton.edu`) so that users can test it before it goes to production.

Deployment uses [cdh-ansible](https://github.com/Princeton-CDH/cdh-ansible). You need a local checkout of `cdh-ansible` with your environment set up — see the [cdh-ansible README](https://github.com/Princeton-CDH/cdh-ansible/blob/main/README.md).

All commands below run from your `cdh-ansible` directory. Staging is the default environment; `develop` is the default branch for staging.

### Deployment gotchas

**Use `-e ref=TAG` to deploy a specific eScriptorium version.**
The default is whatever `gitref` is set to in `group_vars/htr/vars.yml` (currently `v1.0.1`). Pass `-e ref=` to override it — either with a release tag or an unpublished git ref such as a branch name or commit hash.

```sh
# deploy a release tag
ansible-playbook playbooks/escriptorium.yml -e ref=v1.0.1

# deploy an unpublished branch
ansible-playbook playbooks/escriptorium.yml -e ref=dev-0.14.2
```

**Skip the `nfs` tag on new or rebuilt VMs.**
The NFS (TigerData) mount may not be provisioned yet and will block the playbook.

```sh
ansible-playbook playbooks/escriptorium.yml --skip-tags nfs
```

Only run the `nfs` tag after TigerData access has been granted.

**Run `db_backup` explicitly before data-destructive operations.**
The database backup task is disabled by default (controlled by `postgres_backup_enabled: false` in `group_vars/htr/vars.yml`). To run it explicitly before a significant migration or destructive change:

```sh
ansible-playbook playbooks/escriptorium.yml -t db_backup
```

______________________________________________________________________

### Option A: Full redeploy (eScriptorium only)

Use this when eScriptorium itself has changed, or after the first install. Note: this does **not** reinstall htr2hpc — the `reinstall-htr2hpc` tag is tagged `never` and will not run unless explicitly passed.

```sh
ansible-playbook playbooks/escriptorium.yml --skip-tags nfs
```

### Option B: Reinstall htr2hpc only (faster)

Use this when only htr2hpc code has changed, or after any htr2hpc release. Uninstalls and reinstalls the package, then restarts nginx, Celery, and Django Channels. Installs from `htr2hpc_gitref`, which defaults to `develop` on staging.

```sh
ansible-playbook playbooks/escriptorium.yml -t reinstall-htr2hpc
```

To install a specific version:

```sh
ansible-playbook playbooks/escriptorium.yml -t reinstall-htr2hpc -e htr2hpc_gitref=0.7.0
```

### Option C: Deploy both eScriptorium and htr2hpc

When both need updating, combine the options:

```sh
ansible-playbook playbooks/escriptorium.yml --skip-tags nfs -e ref=v1.0.1 -e htr2hpc_gitref=0.7.0 -t all,reinstall-htr2hpc
```

> `ref` and `htr2hpc_gitref` are independent: `ref` controls the eScriptorium git ref; `htr2hpc_gitref` controls the htr2hpc package version.

### Post-deploy checks on staging

After deploying, verify on `test-htr.princeton.edu`:

- The feature works as expected end-to-end
- No errors appear in the application logs
- Supervisor programs are running (celery-main, celery-gpu, celery-live, celery-low-priority, django_channels)
- Confirm with user testers that the feature behaves correctly

### Reverting a staging deploy

If the deploy introduces a regression, you can revert to the previous symlink:

```sh
ansible-playbook playbooks/revert_deploy.yml -e host_group=htr_staging
```

This does **not** reverse Django migrations. If migrations were applied, a manual rollback is needed.

______________________________________________________________________

## After user testing

Once staging testing is complete, follow the release checklist in [DEVELOPERNOTES.md](DEVELOPERNOTES.md#creating-a-new-release) to cut a release and deploy to production.
