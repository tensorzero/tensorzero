# Deploying TensorZero on Kubernetes with Helm

> [!IMPORTANT]
>
> This is a reference deployment setup contributed by the community.
> Feedback and enhancements are welcome!

This example shows how to deploy the TensorZero (including the TensorZero Gateway, the TensorZero UI, and a ClickHouse database) on Kubernetes using Helm.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- Ingress controller installed in your cluster (e.g. `traefik-ingress-controller-v3`)
- StorageClass configured for persistent volumes (e.g. `ebs-gp3-retain`)
- Sufficient resources for running ClickHouse and TensorZero services (recommend at least 4GB memory for minikube)
- If `monitoring.metrics.enabled` is set, [Prometheus Operator](https://prometheus-operator.dev/) needs to be installed in your cluster

## Installing the Chart

To install the chart with the release name `tensorzero`:

```bash
# Create a namespace for tensorzero
kubectl create namespace tensorzero

# Install the chart
helm upgrade --install tensorzero .  -f values.yaml -n tensorzero
```

For local development or testing with minikube, you can use port forwarding to access the services:

```bash
# Port forward the gateway service
kubectl port-forward service/tensorzero-gateway -n tensorzero 3000:3000 &

# Port forward the UI service
kubectl port-forward service/tensorzero-ui -n tensorzero 4000:4000 &
```

### Required Secret Configuration

Before installation, you need to create a secret with the following environment variables:

```bash
kubectl create secret generic tensorzero-secret -n tensorzero \
  --from-literal=TENSORZERO_CLICKHOUSE_URL="http://default:tensorzero@clickhouse-clickhouse.clickhouse.svc.cluster.local:8123" \
  --from-literal=TENSORZERO_GATEWAY_URL="http://tensorzero-gateway.tensorzero.svc.cluster.local:3000" \
  --from-literal=OPENAI_API_KEY="your-openai-api-key"
  # ... include model provider credentials as needed ...
```

> Note: The `TENSORZERO_CLICKHOUSE_URL` and `TENSORZERO_GATEWAY_URL` values are the default values for the TensorZero Gateway and ClickHouse service names. If you have changed the service names, you need to update the secret with the correct values.

## Uninstalling the Chart

To uninstall the `tensorzero` deployment, run:

```bash
helm uninstall tensorzero -n tensorzero
```

## Configuration

The following table lists the configurable parameters of the chart and their default values.

### Gateway Configuration

| Parameter                    | Description                      | Default                         |
| ---------------------------- | -------------------------------- | ------------------------------- |
| `gateway.replicaCount`       | Number of gateway replicas       | `1`                             |
| `gateway.serviceAccountName` | Service account for gateway pods | `""`                            |
| `gateway.image.repository`   | Gateway image repository         | `tensorzero/gateway`            |
| `gateway.image.tag`          | Gateway image tag                | `latest`                        |
| `gateway.image.pullPolicy`   | Gateway image pull policy        | `IfNotPresent`                  |
| `gateway.service.type`       | Gateway service type             | `ClusterIP`                     |
| `gateway.service.port`       | Gateway service port             | `3000`                          |
| `gateway.resources.limits`   | Gateway resource limits          | `cpu: 2000m, memory: 4096Mi`    |
| `gateway.resources.requests` | Gateway resource requests        | `cpu: 2000m, memory: 4096Mi`    |
| `gateway.ingress.enabled`    | Enable gateway ingress           | `true`                          |
| `gateway.ingress.className`  | Gateway ingress class            | `traefik-ingress-controller-v3` |
| `gateway.ingress.hosts`      | Gateway ingress hosts            | `tensorzero-gateway.local`      |

### UI Configuration

| Parameter               | Description                 | Default                         |
|-------------------------|-----------------------------|---------------------------------|
| `ui.deploy`             | Whether to deploy the UI    | `true`                          |
| `ui.replicaCount`       | Number of UI replicas       | `1`                             |
| `ui.serviceAccountName` | Service account for UI pods | `""`                            |
| `ui.image.repository`   | UI image repository         | `tensorzero/ui`                 |
| `ui.image.tag`          | UI image tag                | `latest`                        |
| `ui.image.pullPolicy`   | UI image pull policy        | `IfNotPresent`                  |
| `ui.service.type`       | UI service type             | `ClusterIP`                     |
| `ui.service.port`       | UI service port             | `4000`                          |
| `ui.resources.limits`   | UI resource limits          | `cpu: 1000m, memory: 1024Mi`    |
| `ui.resources.requests` | UI resource requests        | `cpu: 500m, memory: 512Mi`      |
| `ui.ingress.enabled`    | Enable UI ingress           | `true`                          |
| `ui.ingress.className`  | UI ingress class            | `traefik-ingress-controller-v3` |
| `ui.ingress.hosts`      | UI ingress hosts            | `tensorzero-ui.local`           |

### Persistence Configuration

| Parameter                    | Description                | Default                         |
| ---------------------------- | -------------------------- | ------------------------------- |
| `persistence.enabled`        | Enable persistent storage  | `false`                         |
| `persistence.size`           | Storage size               | `10Gi`                          |
| `persistence.accessModes`    | Access modes               | `["ReadWriteOnce"]`             |
| `persistence.storageClass`   | Storage class name         | `""`                            |
| `persistence.mountPath`      | Mount path in containers   | `/app/storage`                  |

### Monitoring Configuration

| Parameter                     | Description                                   | Default |
|-------------------------------|-----------------------------------------------|---------|
| `monitoring.metrics.enabled`  | Enable ServiceMonitor creation                | `false` |
| `monitoring.metrics.interval` | Scrape interval                               | `"30s"` |
| `monitoring.metrics.labels`   | Additional labels to attach to ServiceMonitor | `{}`    |

### ClickHouse Configuration

This chart requires a ClickHouse instance for observability. We recommend using Altinity's ClickHouse Helm chart, which offers better cross-platform support (including ARM64 architecture).

> **Important:** TensorZero doesn't support legacy ClickHouse versions. We recommend using the `altinity/clickhouse-server:24.8.14.10459.altinitystable` image or newer.

To deploy ClickHouse using Altinity's Helm chart:

1. Add the Altinity Helm repository:

   ```bash
   helm repo add altinity https://altinity.github.io/helm-charts
   helm repo update
   ```

2. Deploy a ClickHouse instance:

   ```bash
   # Create a namespace for ClickHouse
   kubectl create namespace clickhouse

   # Install the ClickHouse chart using the provided clickhouse-values.yaml
   # which configures the image version and authentication
   helm install clickhouse altinity/clickhouse -n clickhouse -f clickhouse-values.yaml
   ```

   > Note: The Gateway will automatically create the necessary database when it first connects to ClickHouse.

3. Update your TensorZero values file to disable the built-in ClickHouse and specify the external ClickHouse in your secret:
   ```bash
   kubectl create secret generic tensorzero-secret -n tensorzero \
     --from-literal=TENSORZERO_CLICKHOUSE_URL="http://default:tensorzero@clickhouse-clickhouse.clickhouse.svc.cluster.local:8123/tensorzero" \
     --from-literal=TENSORZERO_GATEWAY_URL="http://tensorzero-gateway.tensorzero.svc.cluster.local:3000" \
     --from-literal=OPENAI_API_KEY="your-openai-api-key"
     # ... include model provider credentials as needed ...
   ```

### `ConfigMap` Configuration

The chart includes a `ConfigMap` with the following default configuration:

- Model configuration for Claude 3.5 Haiku
- Function configuration for chat completions

You can customize the installation by creating a values file `custom-values.yaml`:

```yaml
gateway:
  replicaCount: 2
  resources:
    limits:
      cpu: 4000m
      memory: 8192Mi

ui:
  replicaCount: 2

clickhouse:
  replicaCount: 3
  persistence:
    size: 500Gi
```

Then install with:

```bash
helm install tensorzero ./tensorzero -n tensorzero -f custom-values.yaml
```

## Important Notes

1. The chart requires a secret named `tensorzero-secret` with specific environment variables.
2. In production, never store sensitive data in your version-controlled `values.yaml` file.
3. Make sure your cluster has sufficient resources for the configured replicas and resource limits.
4. The ingress configuration assumes you have a working ingress controller installed.

## Calling the Gateway Endpoint

After successful deployment, you can call the gateway endpoint using curl. Here's an example:

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }'
```

Note: If you're using port forwarding to access the gateway locally, use `http://localhost:3000` as the endpoint. If you're using the ingress, replace with your actual gateway ingress host as configured in the `gateway.ingress.hosts` value.
