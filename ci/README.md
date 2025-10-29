# Github Actions + Merge Queue configuration

## Background

A Github merge queue looks at the required status checks for the repository when determining if a merge queue job passed.
Unfortunately, the required status checks are also used for PR CI (to determine if you can even try to merge the PR).
If we want to require any merge-queue-specific jobs (e.g. tests on an external provider like Buildkite), we need to add a corresponding
'dummy' job for PR CI that always succeeds (which will report the needed status check to Github).

Additionally, since the required status checks is a repository-level setting, it's tricky to add new Github Actions jobs as part
of a PR. Adding the job to the required status checks before the PR merges will cause all other PRS to become stuck
(since the check will never be reported). Without modifying the required status checks, the merge queue will merge the
PR even if the newly-added job fails

## Our approach

We have two 'top-level' statuses: 
* check-all-general-jobs-passed
* merge-checks-buildkite

The `check-all-general-jobs-passed` runs for both PR CI and the merge queue, and depends on all of the other jobs in 'general.yml'.
It reads the job statuses from all of its dependencies, and fails if any of those jobs failed or were cancelled.
In particular, this invokes the live merge-queue-only tests as a nested workflow (run-merge-queue-checks) when we're
inside the merge queue, and skips run-merge-queue-checks when we're in PR ci.

The `merge-checks-buildkite` status is repoted in two places:
* From Buildkite, via an external Github integration
* From `dummy.yml`, via a dummy job that runs in PR CI only, and always succeeds when it *does* run.

This configuration ensures that both PR CI and merge queue jobs will always have statuses reported for 'check-all-general-jobs-passed' and
'merge-checks-buildkite'. We set our required status checks to these two jobs *only*, as these jobs propagate failures from their dependencies. This allows us to control which jobs are required from within an individual PR, without touching the repository-level required status checks.

## Adding a new required CI job

* Create a new job in 'general.yml' (optionally as a nested workflow invocation - see 'run-merge-queue-checks' for an example)
* Add the job name to the 'needs' array in 'check-all-general-jobs-passed' inside 'general.yml'
* If the job should only run as part of the merge queue (not PR CI), add an
  `if: github.repository == 'tensorzero/tensorzero' && github.event_name == 'merge_group'` condition to your job definition

Our 'check-all-general-jobs-passed' job allows jobs to be skipped when running in PR CI, but does not allow *any* skips
when running in the merge queue. As a result, your new job will always be required to pass in the merge queue,
but you're free to customize exactly when it runs in PR CI (e.g. a changed-file filter) to save time and money.

References:
* https://github.com/orgs/community/discussions/103114#discussioncomment-8359045
* https://github.com/orgs/community/discussions/25970
