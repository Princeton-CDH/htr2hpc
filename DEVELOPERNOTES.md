# Developer Notes

We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

## Development Setup

### Check out code

Local development requires a checkout of both htr2hpc and [eScriptorium](https://gitlab.com/scripta/escriptorium). The eScriptorium apps directory must be on your Python path for imports to resolve — it doesn't matter where you clone it, as long as PYTHONPATH points to the right location. For example:

```sh
git clone https://gitlab.com/scripta/escriptorium.git
git clone https://github.com/Princeton-CDH/htr2hpc.git
```

### Set PYTHONPATH

We recommend using [direnv](https://direnv.net/) to set PYTHONPATH automatically when you enter the directory. Create a `.envrc` in your htr2hpc checkout:

```sh
export PYTHONPATH=/path/to/escriptorium/app/apps:/path/to/escriptorium/app
```

Then run `direnv allow`. Alternatively, set the variable manually in your shell profile:

```sh
# bash/zsh
export PYTHONPATH=/path/to/escriptorium/app/apps:/path/to/escriptorium/app

# fish
set -x PYTHONPATH /path/to/escriptorium/app/apps /path/to/escriptorium/app
```

### Install dependencies and setup

#### Method A: Using `devbox` (recommended)

This project uses [devbox](https://www.jetify.com/devbox) to simplify local development setup. Devbox installs the required tools (Python, uv) in an isolated environment without affecting your system, so you can get started with a single command and be confident your setup matches other contributors. If you prefer not to use devbox, you can set up the environment manually — see below.

Install devbox if you don't have it:

```sh
curl -fsSL https://get.jetify.com/devbox | bash
```

Then run `devbox shell` to enter the environment. This installs Python 3.11 and uv via Nix. Run `uv sync --group dev` to install Python dependencies into a local `.venv`. You only need to run `devbox shell` once per terminal session.

Use `devbox run test` to run the test suite from your regular terminal without entering the devbox shell. To verify the environment works without any system dependencies, use `devbox shell --pure`. If you run into unexpected errors, `rm -rf .devbox` usually clears them up.

#### Method B: Without `devbox`

If you prefer not to use devbox, you can set up the environment manually with `uv sync --group dev`.

#### Additional notes

Some eScriptorium dependencies require system libraries (e.g. libvips). See the [eScriptorium full install guide](https://gitlab.com/scripta/escriptorium/-/wikis/full-install) if you encounter errors installing requirements.

### Git flow

We recommend installing git-flow for a better development experience. On OSX, you can install with brew:

```sh
brew install git-flow
```

In your local checkout of htr2hpc code, run `git flow init` to initialize the repository with git-flow and accept all the defaults. (This is a one-time step.)

## Running Tests

Unit tests load eScriptorium's real models (for example `TaskReport` and `OcrModel`) from an eScriptorium checkout, but don't need eScriptorium's full dependencies. Heavy packages the models import, like celery and channels, are stubbed in `tests/test_escriptorium_stubs.py`.

Clone eScriptorium into the htr2hpc checkout (it's gitignored), at the version in `.escriptorium-version`, which is the version CI uses:

```sh
git clone --depth 1 --branch "$(cat .escriptorium-version)" https://gitlab.com/scripta/escriptorium.git escriptorium
```

If you have a checkout of eScriptorium and don't want to move it or create a symbolic link, you can set an environment variable `ESCRIPTORIUM_PATH` with the location.

Tests run on PostgreSQL, like eScriptorium. They connect to `localhost:5432` as user `postgres` with password `postgres` by default; set the `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER` and `POSTGRES_PASSWORD` environment variables to use a different server. One way to run a local server is with Docker:

```sh
docker run -d --name htr2hpc-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
```

Then install htr2hpc with its dependencies (`uv sync` includes the dev dependencies by default) and run the tests:

```sh
uv sync
uv run pytest
```

### Upgrading eScriptorium

The eScriptorium version htr2hpc is tested against is set in one place, `.escriptorium-version`; CI reads it from there. When eScriptorium is upgraded:

1. Update `.escriptorium-version`, and re-clone your local `escriptorium` checkout at the new version.
2. Compare the `escriptorium` dependency group in `pyproject.toml` with the new eScriptorium `requirements.txt`, update the versions, and run `uv lock`.
3. Run the tests. If loading eScriptorium's models fails with `ModuleNotFoundError` for a package the models only import, add it to `STUBBED_MODULES` in `tests/test_escriptorium_stubs.py`.
4. Update the deployed version (`gitref`) in cdh-ansible to match.

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

Once the new release has been merged to `main` and pushed to GitHub, deployment is handled via cdh-ansible. See the [cdh-ansible eScriptorium/htr2hpc application docs](https://github.com/Princeton-CDH/cdh-ansible/blob/main/docs/applications/escriptorium.md) for full deployment instructions, including how to use the `reinstall-htr2hpc` tag to deploy a new htr2hpc version.

Note: this deployment is specific to CDH's Princeton instance of eScriptorium.

CDH's Princeton instance uses the Wagtail content management features. The Ansible playbook must install htr2hpc with the `all` optional dependency group, which includes Wagtail:

```sh
pip install htr2hpc[all]
```

After deploying a new instance for the first time, load the content fixture to pre-populate the Wagtail admin with the default page snippets (the homepage heading and the profile page HPC instructions):

```sh
manage.py loaddata cdh_snippets
```

### Base template copy

When Wagtail is installed, `src/htr2hpc/content/templates/base.html` overrides eScriptorium's `base.html`. It is a copy of the upstream template (eScriptorium v1.0.1) with two blocks added around the body block, `pre_body` and `post_body`, which display content snippets on every page.

**This file must be updated whenever eScriptorium is upgraded:** copy the new upstream `base.html`, then add the `wagtailcore_tags` load and the `pre_body` / `post_body` blocks again. See [docs/design-records/80-content-management.md](docs/design-records/80-content-management.md).

For initial setup after a new deployment, see [Configure Site domain](README.md#configure-site-domain) in the README.
