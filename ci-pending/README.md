# Pending CI workflow

`test.yml` here is the intended Polygram CI workflow but could not be
pushed in the initial commit — the local `gh` Personal Access Token lacks
the `workflow` scope.

## To activate

Either:

1. **Update the PAT scope** — regenerate the gh token with `workflow`
   scope, then `git mv ci-pending/test.yml .github/workflows/test.yml`,
   commit, push.

2. **Add via the GitHub web UI** — paste the contents of `test.yml` into
   `Add file → Create new file` named `.github/workflows/test.yml` on
   `main`, then `git rm ci-pending/test.yml` locally, pull, commit.

After activation this directory can be deleted.
