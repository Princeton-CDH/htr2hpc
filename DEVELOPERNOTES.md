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

Once the new release has been merged to `main` and pushed to GitHub, deployment is handled via cdh-ansible. See the [cdh-ansible escriptorium application docs](https://github.com/Princeton-CDH/cdh-ansible/tree/main/docs/applications) for full deployment instructions.

To update the version of htr2hpc installed on the escriptorium server, use the `reinstall-htr2hpc` tag:

```sh
ansible-playbook playbooks/escriptorium.yml -t reinstall-htr2hpc
```

The `reinstall-htr2hpc` tag triggers a full reinstall of the htr2hpc package in the server environment. Use this whenever you need to update to a new version after a release — it is not needed for other playbook changes.
