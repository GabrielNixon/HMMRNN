# Troubleshooting GitHub Pushes

## Symptom
Merges appear to succeed locally, but pushing the changes to GitHub fails or nothing shows up on the remote repository.

## Likely Cause in This Repository
The current checkout does not define any Git remotes. You can see this by running:

```
git remote -v
```

With the present setup, the command returns no entries, so Git has no destination to send commits. Local merges therefore succeed, but `git push` has nowhere to upload the history, which makes it look like the changes never arrive on GitHub.

## Fix
Add the appropriate GitHub remote and push again. Replace `<repo-url>` with the HTTPS or SSH URL of your GitHub repository.

```
git remote add origin <repo-url>
git push -u origin work
```

If the remote already exists under a different name, use that alias instead of `origin`. Once a remote is configured, future `git push` commands will send your merged commits to GitHub as expected.
