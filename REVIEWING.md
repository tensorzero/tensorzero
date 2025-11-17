# Reviewing PRs at TensorZero

We require a code review of every PR to the repository.
Code review itself is an important part of the engineering process and should allow us to catch issues with correctness, developer experience, and conformance to a spec.
In this document, we list some guidelines (a checklist) to follow when reviewing PRs so that we do not forget to check the necessary boxes.
Reviewers should aim to review internal PRs they are assigned within 1 business day.
We omit changes that will be automatically caught by CI.

## Reviewer checklist

### General

* Does the PR have a clear purpose? Could it be split into more focused changes?
* Does the PR add any dependencies? Is there a good justification for adding them? Are we sure we want to depend on them?

### Specification

* Was a spec written for the change?
    * If not and the change is small: ask for a PR description that covers the change.
    * If not and the change is large: back up and potentially back-port a spec in the original issue.

### Software interfaces

* Does the PR change a public interface (e.g. API endpoints, clients)? If so:
    * Do the interface changes match the spec?
    * Are the names clear and obvious?
    * Is there duplicate functionality?
    * Will the change "box us in" for future functionality?
    * Are there sensible defaults?
* If the change touches the inference API:
    * Does the PR also add the change to the OpenAI compatibility layer, the embedded Python client, and NAPI?
    * Does the PR add tests for the OpenAI client changes in each language we test (Python, Node, Go)?
* If the change touches any CRUD-style API:
    * Does this match the patterns of the `v1/` APIs as much as possible?

### Configuration

Does the PR touch the configuration format? If so:
* If we are breaking an existing interface, are we deprecating gracefully?
* Are the names clear and obvious?
* Is there duplicate functionality?
* Will the change "box us in" for future functionality?
* Are there sensible defaults?
* Is any new config in the most obvious section?

### Data model

Are there changes to anything being written to Postgres or ClickHouse? If so:
* Are there tests covering that what is written can be read as expected?
* Are we taking care not to invalidate historical data? Is this tested?
* Is the new format for writing extensible? (e.g. always tag enums)

### Error Handling

* If an error is because of bad input: is it obvious what was bad and how one might fix it?
* If an error could be fixed by enabling a setting: add instructions on how to resolve it.

### UI

* Are there any large React components? Please try and factor those.
* Are there any new fetches in components? Consider a hook.
* Be extra careful with `useEffect`.

### Testing

#### Coverage

Rust:
* New gateway features that affect calls to providers should exercise the providers.
  The ideal way to do this would be to ask the LLM something that it needs to use the functionality to answer.
  Sometimes this is impossible, but think about if it could be done.
* Any function with nontrivial logic should be unit-tested.

UI:
* New UI features should come with Playwright tests.
* Nontrivial components should have Storybook stories.
* Any tricky code should be unit tested with Vitest.

#### Reliability

* Tests should be [complete and concise](https://abseil.io/resources/swe-book/html/ch12.html#make_your_tests_complete_and_concise) with [descriptive and meaningful](https://testing.googleblog.com/2019/12/testing-on-toilet-tests-too-dry-make.html) code phrases and names.
* Are the tests well-isolated? Will writing another test break them?
* Are the tests going to require a lot of compute? disk? memory?
* As much as possible: can we cache flaky network responses?

#### Consolidation

* Can tests be combined into units that preserve coverage but reduce the overall volume of code? Helpful for saving context and test runtime.

### Performance
* Is the PR adding anything expensive to inference requests, possibly on the critical path?

#### ClickHouse

* Are any joins in ClickHouse strictly necessary?
* Do queries leverage the indexes available?
* Are all filters pushed down as far as possible?
* Can we ensure that data returned is bounded?


### Security

If this PR touches API keys or credentials:
* Take a few minutes and think adversarially about how this could break anything.

### Observability

* If something unexpected fails here, would we be able to look at the logs and understand it?
* Do we need OTel support for this feature?

### Deprecations

Are there any features, configuration settings, or interface options that are being deprecated? If so:
* Are there tests for the new and old versions of the interface?
* Does the old version assert that the deprecation is being properly warned?
* Is there an issue opened for completing the deprecation?
  Ideally, the author should assign to self & set a reminder on their calendar to deal with this.

### Documentation

What is the plan for documenting the new feature?

* At minimum: open an issue to document the feature.
  Write an unpolished explanation of what the change is in the issue so that the documenter has context.
* If the change is not too complex: update the docs inline.
* This can be in a follow-up PR, but we should gate approval on an open issue tracking this.
