# Publishing Helm Charts to ArtifactHub

Our CI pipeline automatically bumps the chart's version and publishes it to [ArtifactHub](https://artifacthub.io/packages/helm/tensorzero/tensorzero) when a new GitHub release is created.
The artifacts are hosted on [Cloudflare R2](https://helm.tensorzero.com).

## Publish the Helm chart manually

To publish the chart manually, you'll need to have `helm` installed.
The steps are:

- Bump the version in Chart.yaml
- Run `helm package .`. This should result in a file `tensorzero-x.y.z.tgz` being written to the current directory.
- Upload the file to our Cloudflare R2 bucket `tensorzero-helm-charts`.
- Generate a new `index.yaml` file by running `helm repo index . --url https://helm.tensorzero.com --merge index.yaml`
- Upload the `index.yaml` file to our Cloudflare R2 bucket `tensorzero-helm-charts`.

At this point, ArtifactHub should scan the bucket within minutes and update the listing.
