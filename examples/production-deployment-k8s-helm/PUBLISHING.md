# Publishing Helm Charts to ArtifactHub
For convenience we publish this Helm chart to ArtifactHub and host the artifact on CloudFlare R2 served at https://helm.tensorzero.com.
When this chart is updated we must update the published artifact as well.
To do this you'll need to have `helm` installed.
The steps are:
 - Bump the version in Chart.yaml
 - Run `helm package .`. This should result in a file `tensorzero-x.y.z.tgz` being written to the current directory.
 - Upload the file to our CloudFlare R2 bucket `tensorzero-helm-charts`.
 - Generate a new `index.yaml` file by running `helm repo index . --url https://helm.tensorzero.com --merge index.yaml`
 - Upload the `index.yaml` file to our CloudFlare R2 bucket `tensorzero-helm-charts`.

At this point, ArtifactHub should scan the bucket within minutes and update the listing.
