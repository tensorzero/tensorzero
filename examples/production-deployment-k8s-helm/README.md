# tensorzero

A Helm chart that deploys the TensorZero platform, including a Gateway service, UI service, and ClickHouse database with optional Zookeeper for high availability.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- Ingress controller installed in your cluster (e.g., traefik-ingress-controller-v3)
- StorageClass configured for persistent volumes (e.g., ebs-gp3-retain)

## Installing the Chart

To install the chart with the release name `tensorzero`:

```bash
# Create a namespace for tensorzero
kubectl create namespace tensorzero

# Install the chart
helm upgrade --install tensorzero .  -f values.yaml -n tensorzero
```

### Required Secret Configuration

Before installation, you need to create a secret with the following environment variables:
```bash
kubectl create secret generic tensorzero-secret -n tensorzero \
  --from-literal=TENSORZERO_CLICKHOUSE_URL="http://clickhouse:8123" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-api-key" \
  --from-literal=TENSORZERO_GATEWAY_URL="http://gateway:80"
```

## Uninstalling the Chart

To uninstall/delete the `tensorzero` deployment:

```bash
helm uninstall tensorzero -n tensorzero
```

## Configuration

The following table lists the configurable parameters of the chart and their default values.

### Gateway Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gateway.replicaCount` | Number of gateway replicas | `1` |
| `gateway.image.repository` | Gateway image repository | `tensorzero/gateway` |
| `gateway.image.tag` | Gateway image tag | `2025.02.6` |
| `gateway.image.pullPolicy` | Gateway image pull policy | `IfNotPresent` |
| `gateway.service.type` | Gateway service type | `ClusterIP` |
| `gateway.service.port` | Gateway service port | `80` |
| `gateway.resources.limits` | Gateway resource limits | `cpu: 2000m, memory: 4096Mi` |
| `gateway.resources.requests` | Gateway resource requests | `cpu: 2000m, memory: 4096Mi` |
| `gateway.ingress.enabled` | Enable gateway ingress | `true` |
| `gateway.ingress.className` | Gateway ingress class | `traefik-ingress-controller-v3` |
| `gateway.ingress.hosts` | Gateway ingress hosts | `tz.example.io` |

### UI Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ui.deploy` | Whether to deploy the UI | `true` |
| `ui.replicaCount` | Number of UI replicas | `1` |
| `ui.image.repository` | UI image repository | `tensorzero/ui` |
| `ui.image.tag` | UI image tag | `2025.02.6` |
| `ui.image.pullPolicy` | UI image pull policy | `IfNotPresent` |
| `ui.service.type` | UI service type | `ClusterIP` |
| `ui.service.port` | UI service port | `4000` |
| `ui.resources.limits` | UI resource limits | `cpu: 1000m, memory: 1024Mi` |
| `ui.resources.requests` | UI resource requests | `cpu: 500m, memory: 512Mi` |
| `ui.ingress.enabled` | Enable UI ingress | `true` |
| `ui.ingress.className` | UI ingress class | `traefik-ingress-controller-v3` |
| `ui.ingress.hosts` | UI ingress hosts | `ui.tz.example.io` |

### ClickHouse Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `clickhouse.deploy` | Whether to deploy ClickHouse | `true` |
| `clickhouse.zookeeper.enabled` | Enable Zookeeper for HA | `true` |
| `clickhouse.zookeeper.persistence.storageClass` | Zookeeper storage class | `ebs-gp3-retain` |
| `clickhouse.zookeeper.persistence.size` | Zookeeper storage size | `50Gi` |
| `clickhouse.auth.username` | ClickHouse username | `default` |
| `clickhouse.auth.password` | ClickHouse password | `tensorzero` |
| `clickhouse.replicaCount` | Number of ClickHouse replicas | `3` |
| `clickhouse.resourcesPreset` | ClickHouse resources preset | `2xlarge` |
| `clickhouse.persistence.storageClass` | ClickHouse storage class | `ebs-gp3-retain` |
| `clickhouse.persistence.size` | ClickHouse storage size | `200Gi` |

### ConfigMap Configuration

The chart includes a ConfigMap with the following default configuration:
- Model configuration for Claude 3.5 Haiku
- Function configuration for chat completions

You can customize the installation by creating a values file:

```yaml
# custom-values.yaml
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
2. In production, never store sensitive data in your version-controlled values.yaml file.
3. The ClickHouse deployment includes Zookeeper for high availability by default.
4. Make sure your cluster has sufficient resources for the configured replicas and resource limits.
5. The ingress configuration assumes you have a working ingress controller installed. 

## Calling the Gateway Endpoint

After successful deployment, you can call the gateway endpoint using curl. Here's an example:

```bash
curl -X POST https://tz-gateway.example.io/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
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

Note: Replace `https://tz-gateway.example.io` with your actual gateway ingress host as configured in the `gateway.ingress.hosts` value. 