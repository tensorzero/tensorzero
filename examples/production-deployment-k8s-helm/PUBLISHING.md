# Publishing Helm Charts to ArtifactHub

Our CI pipeline automatically bumps the chart's version and publishes it to [ArtifactHub](https://artifacthub.io/packages/helm/tensorzero/tensorzero) when a new GitHub release is created.
The artifacts are hosted on [Cloudflare R2](https://helm.tensorzero.com).

## Prerequisites

To contribute changes to the helm chart, you need:

- Helm 3.2.0+ installed [→](https://helm.sh/docs/intro/install/)
- The `helm-values-schema-json` plugin installed:
  ```bash
  helm plugin install https://github.com/losisin/helm-values-schema-json.git
  ```

## Publish the Helm chart manually

To publish the chart manually, you'll need to have `helm` installed.
The steps are:

- Bump the version in Chart.yaml
- Run `helm package .`. This should result in a file `tensorzero-x.y.z.tgz` being written to the current directory.
- Upload the file to our Cloudflare R2 bucket `tensorzero-helm-charts`.
- Generate a new `index.yaml` file by running `helm repo index . --url https://helm.tensorzero.com --merge index.yaml`
- Upload the `index.yaml` file to our Cloudflare R2 bucket `tensorzero-helm-charts`.

At this point, ArtifactHub should scan the bucket within minutes and update the listing.

## Values Schema Generation

The `values.schema.json` file is automatically generated from `values.yaml`, providing Helm chart values validation ([docs](https://helm.sh/docs/topics/charts/#schema-files)).

The schema is automatically generated and validated:

- **Locally**: A pre-commit hook runs when `values.yaml` changes, generating the schema automatically
- **In CI**: The schema is validated to ensure it matches the committed version

If you modify `values.yaml`, the pre-commit hook will regenerate `values.schema.json`. Make sure to commit the updated schema file along with your changes.
