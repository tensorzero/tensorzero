# Github Actions + Merge Queue configuration

## Background

A Github merge queue looks at the required status checks for the repository when determining if a merge queue job passed.
Unfortunately, the required status checks are also used for PR CI (to determine if you can even try to merge the PR).
If we want to require any merge-queue-specific jobs (e.g. live tests against providers), we need to add a corresponding
'dummy' job for PR CI that always succeeds (which will report the needed status check to Github).

Additionally, since the required status checks is a repository-level setting, it's tricky to add new Github Actions jobs as part
of a PR. Adding the job to the required status checks before the PR merges will cause all other PRS to become stuck
(since the check will never be reported). Without modifying the required status checks, the merge queue will merge the
PR even if the newly-added job fails

## Our approach

We have two 'top-level' jobs:
* check-all-general-jobs-passed
* check-all-live-tests-passed

The `check-all-general-jobs-passed` runs for both PR CI and the merge queue, and depends on all of the other jobs in 'general.yml'.
It reads the job statuses from all of its dependencies, and fails if any of those jobs failed or were cancelled.

The `check-all-live-tests-passed` is similar, except we have two versions of it.
In `merge-queue.yml`, it depends on the `live-tests` job. In `dummy.yml`, it's a dummy job that has no dependencies and always succeeds (and runs in PR CI only)

This configuration ensures that both PR CI and merge queue jobs will always have statuses reported for 'check-all-general-jobs-passed' and
'check-all-live-tests-passed'. We set our required status checks to these two jobs *only*, as these jobs propagate failures from their dependencies. This allows us to control which jobs are required from within an individual PR, without touching the repository-level required status checks.

## Adding a new required CI job

If you want the job to run in both PR CI and merge queue jobs:
* Add a new job to 'general.yml'
* Add the job name to the 'needs' array of 'check-all-general-jobs-passed' in 'general.yml'

If you want the job to run only for merge queue jobs (not PR CI):
* Add a new job to 'merge-queue.yml'
* Add the job name to the 'needs' array of 'check-all-live-tests-passed' in 'merge-queue.yml'. You do *not* need to modify 'dummy.yml' or 'general.yml'


References:
* https://github.com/orgs/community/discussions/103114#discussioncomment-8359045
* https://github.com/orgs/community/discussions/25970
