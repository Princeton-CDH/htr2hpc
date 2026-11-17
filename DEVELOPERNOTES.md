# Developer Notes

We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

## Development Setup

### Prerequisites

- [git](https://git-scm.com/)
- [Docker](https://docs.docker.com/get-docker/), or any PostgreSQL 15 server — the tests run on PostgreSQL, like eScriptorium
- [devbox](https://www.jetify.com/devbox) (recommended) or [uv](https://docs.astral.sh/uv/getting-started/installation/); see the install options below. You don't need to install Python yourself, since uv installs the version listed in `.python-version`.

### Check out code

Clone htr2hpc and switch to the `develop` branch, where development happens (`main` is the most recent release). Then clone the [eScriptorium](https://gitlab.com/scripta/escriptorium) version we test against (listed in `.escriptorium-version`) into an `escriptorium` directory inside the htr2hpc checkout; it is gitignored.

```sh
git clone https://github.com/Princeton-CDH/htr2hpc.git
cd htr2hpc
git checkout develop
git clone --depth 1 --branch "$(cat .escriptorium-version)" https://gitlab.com/scripta/escriptorium.git escriptorium
```

The tests load eScriptorium's real models from this checkout, but they don't need eScriptorium's full dependencies. The tests add the checkout to `sys.path` themselves (see `tests/test_escriptorium_stubs.py`), so you don't need to set `PYTHONPATH`. If you already have an eScriptorium checkout and don't want to move it or create a symbolic link, set the `ESCRIPTORIUM_PATH` environment variable to its location.

### Install dependencies and setup

#### Method A: Using `devbox` (recommended)

This project uses [devbox](https://www.jetify.com/devbox) to simplify local development setup. Devbox installs the required tools (Python, uv) in an isolated environment without affecting your system, so you can get started with a single command and be confident your setup matches other contributors. If you prefer not to use devbox, you can set up the environment manually — see below.

Install devbox if you don't have it:

```sh
curl -fsSL https://get.jetify.com/devbox | bash
```

Then run `devbox shell` to enter the environment. This installs uv via Nix; uv then installs the Python version listed in `.python-version`. Run `uv sync` to install Python dependencies into a local `.venv`; the `dev` group is installed by default and includes everything for development, including the optional Wagtail dependency. You only need to run `devbox shell` once per terminal session.

Use `devbox run test` to run the test suite from your regular terminal without entering the devbox shell. To verify the environment works without any system dependencies, use `devbox shell --pure`. If you run into unexpected errors, `rm -rf .devbox` usually clears them up.

#### Method B: Without `devbox`

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run `uv sync` to install Python (the version in `.python-version`) and the dependencies into a local `.venv`.

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run linting and formatting checks before each commit. It is included in the `dev` dependency group, so `uv sync` installs it. Install the git hook once per checkout:

```sh
uv run pre-commit install
```

The hooks are configured in `.pre-commit-config.yaml` and include ruff (lint and format), mdformat, yamlfmt, codespell, a `uv.lock` check, and GitHub Actions workflow validation. On each commit, the hooks check only the files you changed. Hooks that fix files (ruff, mdformat, yamlfmt) will modify them and fail the commit; review the changes, `git add` them, and commit again.

To run the hooks on the files you have staged without committing:

```sh
uv run pre-commit run
```

Avoid `pre-commit run --all-files` unless you are doing a formatting cleanup: parts of the existing codebase are not yet formatted to match the hooks, so it will modify files you didn't change.

### Git flow

We recommend installing git-flow for a better development experience. On OSX, you can install with brew:

```sh
brew install git-flow
```

In your local checkout of htr2hpc code, run `git flow init` to initialize the repository with git-flow and accept all the defaults. (This is a one-time step.)

### Making changes

Start from `develop`: create a branch with `git flow feature start <name>` (or `git checkout -b <name> develop`), and open pull requests against `develop`. For user-visible changes, add an entry to `CHANGELOG.md`; CI fails a pull request that doesn't change it. For changes that aren't user-visible, ask a maintainer to add the `no changelog` label to the pull request.

## Running Tests

Start a PostgreSQL server. Tests connect to `localhost:5432` as user `postgres` with password `postgres` by default. One way to run a local server is with Docker:

```sh
docker run -d --name htr2hpc-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
```

If port 5432 is already in use on your machine, map a different port (for example `-p 5433:5432`) and set `POSTGRES_PORT=5433` when running the tests. The `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER` and `POSTGRES_PASSWORD` environment variables let you use any other server.

Then install htr2hpc with its dependencies (`uv sync` includes the dev dependencies by default, including Wagtail) and run the tests:

```sh
uv sync
uv run pytest
```

Inside a devbox shell, or from your regular terminal, `devbox run test` runs the same test suite.

CI runs the tests twice, with and without Wagtail, to make sure it stays optional; the content management tests are skipped when Wagtail isn't installed. To run the tests without Wagtail locally, install only the `test` group:

```sh
uv sync --no-dev --group test
uv run --no-sync pytest
```

### Upgrading eScriptorium

The eScriptorium version htr2hpc is tested against is set in one place, `.escriptorium-version`; CI reads it from there. When eScriptorium is upgraded:

1. Update `.escriptorium-version`, and re-clone your local `escriptorium` checkout at the new version.
2. Compare the `escriptorium` dependency group in `pyproject.toml` with the new eScriptorium `requirements.txt`, update the versions, and run `uv lock`.
3. Run the tests. If loading eScriptorium's models fails with `ModuleNotFoundError` for a package the models only import, add it to `STUBBED_MODULES` in `tests/test_escriptorium_stubs.py`.
4. Update the deployed version (`gitref`) in cdh-ansible to match.

### Changing the Python version

The Python version used for development and CI is set in `.python-version`; `uv` (including in devbox and CI) reads it from there. It should match the version eScriptorium runs on the servers. Read the Docs doesn't read `.python-version`, so when you change it, also update `build.tools.python` in `.readthedocs.yaml`.

`requires-python` in `pyproject.toml` is the minimum supported version, not the version we test on, so it only changes when we drop support for an older version. The HPC environment (`src/htr2hpc/train/user_setup.sh`) uses its own Python version, set by kraken's requirements.

## Building Documentation

Building documentation alone doesn't require eScriptorium or its dependencies to be installed.

Install htr2hpc with docs dependencies, then run:

```sh
uv sync --group docs
cd sphinx-docs && uv run make html
```

The built documentation will be in `sphinx-docs/_build/html/`. Documentation is also published automatically to [ReadTheDocs](https://htr2hpc.readthedocs.io/) on every push to `main`.

## Creating a new release

Follow the release checklist in the GitHub issue template for full release prep steps, including acceptance testing and changelog review.

Release tags must use the `X.Y.Z` format without a `v` prefix (e.g. `0.9.0`, not `v0.9.0`). Both `user_setup.sh` and the Django-side `ensure_htr2hpc_version()` use the version string directly in the pip install URL. When `HTR2HPC_GITREF` is not set (production), `ensure_htr2hpc_version()` falls back to `__version__`, so the tag must match `__version__` exactly or pip will fail with a pathspec error. Staging deployments set `HTR2HPC_GITREF` to a commit SHA and are not affected by this constraint.

## Deploying a new release

Once the new release has been merged to `main` and pushed to GitHub, deployment is handled via cdh-ansible. See the [cdh-ansible eScriptorium/htr2hpc application docs](https://github.com/Princeton-CDH/cdh-ansible/blob/main/docs/applications/escriptorium.md) for full deployment instructions. Every run of the playbook reinstalls htr2hpc from the git ref set with `-e htr2hpc_gitref=...` (release tag or branch), so there is no separate step for updating htr2hpc.

Note: this deployment is specific to CDH's Princeton instance of eScriptorium.

CDH's Princeton instance uses the Wagtail content management features, so cdh-ansible installs htr2hpc with the `all` optional dependency group (`htr2hpc[all]`), which includes Wagtail.

After deploying a new instance for the first time, load the content fixture to pre-populate the Wagtail admin with the default page snippets (the homepage heading and the profile page HPC instructions):

```sh
manage.py loaddata cdh_snippets
```

### Base template copy

When Wagtail is installed, `src/htr2hpc/content/templates/base.html` overrides eScriptorium's `base.html`. It is a copy of the upstream template (eScriptorium v1.0.1) with two blocks added around the body block, `pre_body` and `post_body`, which display content snippets on every page.

**This file must be updated whenever eScriptorium is upgraded:** copy the new upstream `base.html`, then add the `wagtailcore_tags` load and the `pre_body` / `post_body` blocks again. See [docs/design-records/80-content-management.md](docs/design-records/80-content-management.md).

For initial setup after a new deployment, see [Configure Site domain](README.md#configure-site-domain) in the README.
