# Deploy Notes


We use the [git-flow branching pattern](https://www.gitkraken.com/learn/git/git-flow) for this codebase. This means that `main` is always the most recent release and `develop` has new features for the next release.

We recommend installing git-flow. On OSX, you can install with brew:
```sh
brew install git-flow
```

In your local checkout of htr2hpc code, run `git flow init` to initialize the repository with git-flow and accept all the defaults. (This is a one-time step.)

## Creating a new release

Use git flow to create a branch to prep a new release. Specify the version number of the new release you are planning to create; e.g., for version 0.5:

```sh
git flow release start 0.5
```

1. Update the version number in `src/htr2hpc/__init__.py`
2. Update `CHANGELOG.md` to document changes in the new version.

Do any checking you want to do to verify the changes (e.g., git diff against `main` or previous release, or creating a pull request on GitHub to review the changes).

When you are done making changes, you can use git flow to finish the release:

```sh
git flow release finish 0.5
```

This will merge your changes into `main`, `develop`, and create a new tag for your version.

When you create the tag, you can add a brief message describing the release, e.g.:

```
v0.5 - initial release for beta testing
```

Push all the changes to GitHub:

```sh
git checkout main
git push
git checkout develop
git push
git push --tags
```

## Deploying a new release

Once the new release has been merged to `main` and pushed to GitHub, you can use the cdh-ansible playbook to update the version of htr2hpc installed on the escriptorium test server using the *htr2hpc-reinstall* tag:

```sh
ansible-playbook playbooks/escriptorium.yml -t reinstall-htr2hpc
```

