# Developer Notes

We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

## Development Setup

This project uses [devbox](https://www.jetify.com/devbox) to provide a reproducible development environment with Python 3.11 and uv pre-installed.

### Install devbox

```sh
curl -fsSL https://get.jetify.com/devbox | bash
```

### Start the development environment

```sh
devbox shell
```

This installs the required packages via Nix and runs `uv sync` to install Python dependencies into a local `.venv`. You only need to run `devbox shell` once per terminal session.

### Available scripts

| Command | Description |
| --- | --- |
| `devbox run test` | Run the test suite with pytest |
| `devbox run lint` | Check code with ruff |
| `devbox run format` | Format code with ruff |
| `devbox run typecheck` | Run mypy type checking |

### Verify the environment

To test that the devbox environment works without relying on any system dependencies:

```sh
devbox shell --pure
```

This launches a shell with only the packages specified in `devbox.json`, which is useful for confirming the environment is self-contained. You can then run the tests:

```sh
devbox run test
```

If you run into unexpected errors, clearing devbox's local state usually fixes them:

```sh
rm -rf .devbox
```

### Without devbox

If you prefer not to use devbox, you can set up the environment manually with uv:

```sh
uv sync
```

### git-flow

We recommend installing git-flow. On OSX, you can install with brew:

```sh
brew install git-flow
```

In your local checkout of htr2hpc code, run `git flow init` to initialize the repository with git-flow and accept all the defaults. (This is a one-time step.)

## Creating a new release

Follow the release checklist in the GitHub issue template for full release prep steps, including acceptance testing and changelog review.

## Deploying a new release

Once the new release has been merged to `main` and pushed to GitHub, deployment is handled via cdh-ansible. See the [cdh-ansible eScriptorium/htr2hpc application docs](https://github.com/Princeton-CDH/cdh-ansible/blob/main/docs/applications/escriptorium.md) for full deployment instructions, including how to use the `reinstall-htr2hpc` tag to deploy a new htr2hpc version.

Note: this deployment is specific to CDH's Princeton instance of eScriptorium.
