# Developer Notes

We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

## Development Setup

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
