# Azure OpenAI Embeddings with TensorZero

This example demonstrates how to use Azure OpenAI embeddings through TensorZero, showcasing both the TensorZero client and the OpenAI SDK with TensorZero's patch functionality.

## Overview

Azure OpenAI provides enterprise-grade AI services with enhanced security, compliance, and regional availability. TensorZero integrates seamlessly with Azure OpenAI deployments, offering:

- Unified API across multiple Azure regions
- Automatic failover and load balancing
- Enterprise security and compliance features
- Detailed usage tracking and observability

## Prerequisites

1. **Python 3.8+**
2. **Azure OpenAI Resource**: Create an Azure OpenAI resource in the Azure portal
3. **Embedding Model Deployments**: Deploy embedding models in your Azure OpenAI resource
4. **API Key**: Get your API key from the Azure portal

## Setup

### 1. Create Azure OpenAI Resource

1. Go to the [Azure Portal](https://portal.azure.com/)
2. Create a new "Azure OpenAI" resource
3. Choose your region and pricing tier
4. Wait for deployment to complete

### 2. Deploy Embedding Models

1. Go to your Azure OpenAI resource
2. Navigate to "Model deployments"
3. Click "Create new deployment"
4. Deploy the following models:
   - `text-embedding-3-small` (recommend deployment name: `text-embedding-3-small`)
   - `text-embedding-3-large` (recommend deployment name: `text-embedding-3-large`)
   - `text-embedding-ada-002` (recommend deployment name: `text-embedding-ada-002`)

### 3. Configure TensorZero

1. **Update the configuration file** `config/tensorzero.toml`:

   ```toml
   [embedding_models.text-embedding-3-small.providers.azure]
   type = "azure"
   endpoint = "https://your-resource-name.openai.azure.com/"
   deployment_id = "your-deployment-name"
   ```

   Replace:
   - `your-resource-name` with your Azure OpenAI resource name
   - `your-deployment-name` with your deployment name

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key:**

   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the Python example:**

   ```bash
   python main.py
   ```

5. **Run the Node.js example (optional):**

   ```bash
   npm install
   npm start
   ```

## What This Example Shows

### Method 1: TensorZero Client (Recommended)

```python
async with AsyncTensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml"
) as client:
    response = await client.embed(
        model_name="text-embedding-3-small",
        input=texts
    )
```

### Method 2: OpenAI SDK with TensorZero Patch

```python
client = AsyncOpenAI()
client = await patch_openai_client(
    client,
    config_file="config/tensorzero.toml"
)

response = await client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-small"
)
```

## Configuration Details

The `config/tensorzero.toml` file configures Azure OpenAI embedding models:

```toml
[embedding_models.text-embedding-3-small]
routing = ["azure"]

[embedding_models.text-embedding-3-small.providers.azure]
type = "azure"
endpoint = "https://your-resource-name.openai.azure.com/"
deployment_id = "text-embedding-3-small"
```

### Key Configuration Parameters

- **endpoint**: Your Azure OpenAI resource endpoint
- **deployment_id**: The name you gave your model deployment
- **routing**: Can include multiple providers for failover

## Features Demonstrated

- **Single text embedding**: Generate embeddings for individual texts
- **Batch embedding**: Process multiple texts in a single request
- **Multiple models**: Switch between different embedding models
- **Custom dimensions**: Specify embedding dimensions (where supported)
- **Token usage tracking**: Monitor Azure OpenAI usage
- **Error handling**: Graceful handling of deployment-specific limitations

## Azure-Specific Considerations

### Regional Deployment

- Deploy in multiple regions for redundancy
- Choose regions close to your users for lower latency
- Some regions may have different model availability

### Pricing and Quotas

- Azure OpenAI charges per 1K tokens
- Monitor usage through the Azure portal
- Set up quotas and alerts to control costs

### Security and Compliance

- Azure OpenAI offers enhanced security features
- Private endpoints for secure network access
- Compliance with enterprise security standards
- Data residency guarantees

## Advanced Configuration

### Multi-Region Failover

```toml
[embedding_models.text-embedding-3-small]
routing = ["azure-east", "azure-west"]

[embedding_models.text-embedding-3-small.providers.azure-east]
type = "azure"
endpoint = "https://resource-east.openai.azure.com/"
deployment_id = "text-embedding-3-small"

[embedding_models.text-embedding-3-small.providers.azure-west]
type = "azure"
endpoint = "https://resource-west.openai.azure.com/"
deployment_id = "text-embedding-3-small"
```

### Custom Timeouts

```toml
[embedding_models.text-embedding-3-small.providers.azure]
type = "azure"
endpoint = "https://your-resource.openai.azure.com/"
deployment_id = "text-embedding-3-small"

[embedding_models.text-embedding-3-small.providers.azure.timeouts]
non_streaming = { total_ms = 30000 }  # 30 second timeout
```

## Troubleshooting

### Common Issues

**"Resource not found" error:**
- Verify your endpoint URL is correct
- Ensure the deployment exists and is active

**"Insufficient quota" error:**
- Check your quota limits in the Azure portal
- Request quota increases if needed

**"Authentication failed" error:**
- Verify your API key is correct
- Check that the key hasn't expired

### Getting Help

1. Check the Azure portal for deployment status
2. Review Azure OpenAI service logs
3. Verify network connectivity to your endpoint
4. Ensure your deployment has sufficient quota

## Next Steps

- Deploy models in multiple Azure regions for redundancy
- Set up monitoring and alerting in Azure
- Integrate with your enterprise authentication system
- Explore Azure's private endpoint options for enhanced security

## Learn More

- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [TensorZero Documentation](https://www.tensorzero.com/docs)
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
- [Azure OpenAI Quotas and Limits](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/quotas-limits)