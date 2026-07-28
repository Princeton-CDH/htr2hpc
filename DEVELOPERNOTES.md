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

Running unit tests does not require eScriptorium or its dependencies to be installed.

Install htr2hpc with dev dependencies (which include test dependencies), then run:

```sh
uv sync --group dev
uv run pytest
```

## Building Documentation

Building documentation alone doesn't require eScriptorium installed.

Install htr2hpc with docs dependencies, then run:

```sh
uv sync --group docs
cd sphinx-docs && uv run make html
```

The built documentation will be in `sphinx-docs/_build/html/`. Documentation is also published automatically to [ReadTheDocs](https://htr2hpc.readthedocs.io/) on every push to `main`.


## Creating a new release

Follow the release checklist in the GitHub issue template for full release prep steps, including acceptance testing and changelog review.

## Deploying a new release

Once the new release has been merged to `main` and pushed to GitHub, deployment is handled via cdh-ansible. See the [cdh-ansible eScriptorium/htr2hpc application docs](https://github.com/Princeton-CDH/cdh-ansible/blob/main/docs/applications/escriptorium.md) for full deployment instructions, including how to use the `reinstall-htr2hpc` tag to deploy a new htr2hpc version.

Note: this deployment is specific to CDH's Princeton instance of eScriptorium.

For initial setup after a new deployment, see [Configure Site domain](README.md#configure-site-domain) in the README.
