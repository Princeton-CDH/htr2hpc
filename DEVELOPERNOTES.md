# Developer Notes

We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

## Development Setup

This project uses [devbox](https://www.jetify.com/devbox) to simplify local development setup. Devbox installs the required tools (Python, uv) in an isolated environment without affecting your system, so you can get started with a single command and be confident your setup matches other contributors. If you prefer not to use devbox, you can set up the environment manually — see below.

Install devbox if you don't have it:

```sh
curl -fsSL https://get.jetify.com/devbox | bash
```

Then run `devbox shell` to enter the environment. This installs Python 3.11 and uv via Nix. Run `uv sync` to install Python dependencies into a local `.venv`. You only need to run `devbox shell` once per terminal session.

Use `devbox run test` to run the test suite from your regular terminal without entering the devbox shell. To verify the environment works without any system dependencies, use `devbox shell --pure`. If you run into unexpected errors, `rm -rf .devbox` usually clears them up.

If you prefer not to use devbox, you can set up the environment manually with `uv sync`.

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
