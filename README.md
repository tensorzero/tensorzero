<p><picture><img src="https://github.com/user-attachments/assets/47d67430-386d-4675-82ad-d4734d3262d9" alt="TensorZero Logo" width="128" height="128"></picture></p>

# TensorZero

<p><picture><img src="https://www.tensorzero.com/github-trending-badge.svg" alt="#1 Repository Of The Day"></picture></p>

**TensorZero is an open-source stack for _industrial-grade LLM applications_:**

- **Gateway:** access every LLM provider through a unified API, built for performance (<1ms p99 latency)
- **Observability:** store inferences and feedback in your database, available programmatically or in the UI
- **Optimization:** collect metrics and human feedback to optimize prompts, models, and inference strategies
- **Evaluation:** benchmark individual inferences or end-to-end workflows using heuristics, LLM judges, etc.
- **Experimentation:** ship with confidence with built-in A/B testing, routing, fallbacks, retries, etc.

Take what you need, adopt incrementally, and complement with other tools.

<video src="https://github.com/user-attachments/assets/04a8466e-27d8-4189-b305-e7cecb6881ee"></video>

---

<p align="center">
  <b><a href="https://www.tensorzero.com/" target="_blank">Website</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs" target="_blank">Docs</a></b>
  ·
  <b><a href="https://www.x.com/tensorzero" target="_blank">Twitter</a></b>
  ·
  <b><a href="https://www.tensorzero.com/slack" target="_blank">Slack</a></b>
  ·
  <b><a href="https://www.tensorzero.com/discord" target="_blank">Discord</a></b>
  <br>
  <br>
  <b><a href="https://www.tensorzero.com/docs/quickstart" target="_blank">Quick Start (5min)</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Deployment Guide</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/api-reference" target="_blank">API Reference</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Configuration Reference</a></b>
</p>

---

## Features

### 🌐 LLM Gateway

> **Integrate with TensorZero once and access every major LLM provider.**

- [x] **[Call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm)** (API or self-hosted) through a single unified API
- [x] Infer with **[streaming](https://www.tensorzero.com/docs/gateway/guides/streaming-inference)**, **[tool use](https://www.tensorzero.com/docs/gateway/guides/tool-use)**, structured generation, **[batch](https://www.tensorzero.com/docs/gateway/guides/batch-inference)**, **[embeddings](https://www.tensorzero.com/docs/gateway/generate-embeddings)**, **[multimodal (images, files)](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)**, **[caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching)**, etc.
- [x] **[Create prompt templates and schemas](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)** to enforce a consistent, typed interface between your application and the LLMs
- [x] Satisfy extreme throughput and latency needs, thanks to 🦀 Rust: **[<1ms p99 latency overhead at 10k+ QPS](https://www.tensorzero.com/docs/gateway/benchmarks)**
- [x] Use any programming language: **[integrate via our Python client, any OpenAI SDK, or our HTTP API](https://www.tensorzero.com/docs/gateway/clients)**
- [x] **[Ensure high availability](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)** with routing, retries, fallbacks, load balancing, granular timeouts, etc.
- [x] **[Enforce custom rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits)** with granular scopes (e.g. user-defined tags) to keep usage under control
- [ ] Soon: spend tracking and budgeting, service accounts

<br>

**Supported Model Providers:**
**[Anthropic](https://www.tensorzero.com/docs/gateway/guides/providers/anthropic)**,
**[AWS Bedrock](https://www.tensorzero.com/docs/gateway/guides/providers/aws-bedrock)**,
**[AWS SageMaker](https://www.tensorzero.com/docs/gateway/guides/providers/aws-sagemaker)**,
**[Azure OpenAI Service](https://www.tensorzero.com/docs/gateway/guides/providers/azure)**,
**[DeepSeek](https://www.tensorzero.com/docs/gateway/guides/providers/deepseek)**,
**[Fireworks](https://www.tensorzero.com/docs/gateway/guides/providers/fireworks)**,
**[GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-anthropic)**,
**[GCP Vertex AI Gemini](https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-gemini)**,
**[Google AI Studio (Gemini API)](https://www.tensorzero.com/docs/gateway/guides/providers/google-ai-studio-gemini)**,
**[Groq](https://www.tensorzero.com/docs/gateway/guides/providers/groq)**,
**[Hyperbolic](https://www.tensorzero.com/docs/gateway/guides/providers/hyperbolic)**,
**[Mistral](https://www.tensorzero.com/docs/gateway/guides/providers/mistral)**,
**[OpenAI](https://www.tensorzero.com/docs/gateway/guides/providers/openai)**,
**[OpenRouter](https://www.tensorzero.com/docs/gateway/guides/providers/openrouter)**,
**[SGLang](https://www.tensorzero.com/docs/gateway/guides/providers/sglang)**,
**[TGI](https://www.tensorzero.com/docs/gateway/guides/providers/tgi)**,
**[Together AI](https://www.tensorzero.com/docs/gateway/guides/providers/together)**,
**[vLLM](https://www.tensorzero.com/docs/gateway/guides/providers/vllm)**, and
**[xAI (Grok)](https://www.tensorzero.com/docs/gateway/guides/providers/xai)**.
Need something else? TensorZero also supports **[any OpenAI-compatible API (e.g. Ollama)](https://www.tensorzero.com/docs/gateway/guides/providers/openai-compatible)**.

<br>

<details open>
<summary><b>Usage: Python &mdash; TensorZero Client (Recommended)</b></summary>

You can access any provider using the TensorZero Python client.

1. `pip install tensorzero`
2. Optional: Set up the TensorZero configuration.
3. Run inference:

```python
from tensorzero import TensorZeroGateway  # or AsyncTensorZeroGateway


with TensorZeroGateway.build_embedded(clickhouse_url="...", config_file="...") as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        # Try other providers easily: "anthropic::claude-3-7-sonnet-20250219"
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ]
        },
    )
```

See **[Quick Start](https://www.tensorzero.com/docs/quickstart)** for more information.

</details>

<details>
<summary><b>Usage: Python &mdash; OpenAI SDK</b></summary>

You can access any provider using the OpenAI Python SDK with TensorZero.

1. `pip install tensorzero`
2. Optional: Set up the TensorZero configuration.
3. Run inference:

```python
from openai import OpenAI  # or AsyncOpenAI
from tensorzero import patch_openai_client

client = OpenAI()

patch_openai_client(
    client,
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    config_file="config/tensorzero.toml",
    async_setup=False,
)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-4o-mini",
    # Try other providers easily: "tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219"
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about artificial intelligence.",
        }
    ],
)
```

See **[Quick Start](https://www.tensorzero.com/docs/quickstart)** for more information.

</details>

<details>
<summary><b>Usage: JavaScript / TypeScript (Node) &mdash; OpenAI SDK</b></summary>

You can access any provider using the OpenAI Node SDK with TensorZero.

1. Deploy `tensorzero/gateway` using Docker.
   **[Detailed instructions →](https://www.tensorzero.com/docs/gateway/deployment)**
2. Set up the TensorZero configuration.
3. Run inference:

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::model_name::openai::gpt-4o-mini",
  // Try other providers easily: "tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219"
  messages: [
    {
      role: "user",
      content: "Write a haiku about artificial intelligence.",
    },
  ],
});
```

See **[Quick Start](https://www.tensorzero.com/docs/quickstart)** for more information.

</details>

<details>
<summary><b>Usage: Other Languages & Platforms &mdash; HTTP API</b></summary>

TensorZero supports virtually any programming language or platform via its HTTP API.

1. Deploy `tensorzero/gateway` using Docker.
   **[Detailed instructions →](https://www.tensorzero.com/docs/gateway/deployment)**
2. Optional: Set up the TensorZero configuration.
3. Run inference:

```bash
curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Write a haiku about artificial intelligence."
        }
      ]
    }
  }'
```

See **[Quick Start](https://www.tensorzero.com/docs/quickstart)** for more information.

</details>

<br>

### 🔍 LLM Observability

> **Zoom in to debug individual API calls, or zoom out to monitor metrics across models and prompts over time &mdash; all using the open-source TensorZero UI.**

- [x] Store inferences and **[feedback (metrics, human edits, etc.)](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback)** in your own database
- [x] Dive into individual inferences or high-level aggregate patterns using the TensorZero UI or programmatically
- [x] **[Build datasets](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints)** for optimization, evaluation, and other workflows
- [x] Replay historical inferences with new prompts, models, inference strategies, etc.
- [x] **[Export OpenTelemetry traces (OTLP)](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces)** and **[export Prometheus metrics](https://www.tensorzero.com/docs/observability/export-prometheus-metrics)** to your favorite application observability tools
- [ ] Soon: AI-assisted debugging and root cause analysis; AI-assisted data labeling

<table>
<tr></tr> <!-- flip highlight order -->
<tr>
<td width="50%" align="center" valign="middle"><b>Observability » UI</b></td>
<td width="50%" align="center" valign="middle"><b>Observability » Programmatic</b></td>
</tr>
<tr>
<td width="50%" align="center" valign="middle"><video src="https://github.com/user-attachments/assets/a23e4c95-18fa-482c-8423-6078fb4cf285"></video></td>
<td width="50%" align="left" valign="middle">

```python
t0.experimental_list_inferences(
  function_name="sales_agent",
  variant_name="qwen3-promptv2",
  filters=BooleanMetricFilter(
      metric_name="converted_sale",
      value=True,
  ),
  order_by=[OrderBy(by="timestamp", direction="DESC")],
  limit=100_000,
  # ... and more ...
)
```

</td>
</tr>
</table>

<br>

### 📈 LLM Optimization

> **Send production metrics and human feedback to easily optimize your prompts, models, and inference strategies &mdash; using the UI or programmatically.**

- [x] Optimize your models with supervised fine-tuning, RLHF, and other techniques
- [x] Optimize your prompts with automated prompt engineering algorithms like MIPROv2
- [x] Optimize your inference strategy with dynamic in-context learning, chain of thought, best/mixture-of-N sampling, etc.
- [x] Enable a feedback loop for your LLMs: a data & learning flywheel turning production data into smarter, faster, and cheaper models
- [ ] Soon: synthetic data generation

#### Model Optimization

Optimize closed-source and open-source models using supervised fine-tuning (SFT) and preference fine-tuning (DPO).

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b>Supervised Fine-tuning &mdash; UI</b></td>
    <td width="50%" align="center" valign="middle"><b>Preference Fine-tuning (DPO) &mdash; Jupyter Notebook</b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><video src="https://github.com/user-attachments/assets/82f76be7-5e02-4ada-b503-69dfa209a442"></video></td>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/a67a0634-04a7-42b0-b934-9130cb7cdf51"></td>
  </tr>
</table>

#### Inference-Time Optimization

Boost performance by dynamically updating your prompts with relevant examples, combining responses from multiple inferences, and more.

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#best-of-n-sampling">Best-of-N Sampling</a></b></td>
    <td width="50%" align="center" valign="middle"><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#mixture-of-n-sampling">Mixture-of-N Sampling</a></b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/c0edfa4c-713c-4996-9964-50c0d26e6970"></td>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/75b5bf05-4c1f-43c4-b158-d69d1b8d05be"></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#dynamic-in-context-learning-dicl">Dynamic In-Context Learning (DICL)</a></b></td>
    <td width="50%" align="center" valign="middle"><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#chain-of-thought-cot">Chain-of-Thought (CoT)</a></b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/d8489e92-ce93-46ac-9aab-289ce19bb67d"></td>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/ea13d73c-76a4-4e0c-a35b-0c648f898311" height="320"></td>
  </tr>
</table>

_More coming soon..._

<br>

#### Prompt Optimization

Optimize your prompts programmatically using research-driven optimization techniques.

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#best-of-n-sampling">MIPROv2</a></b></td>
    <td width="50%" align="center" valign="middle"><b><a href="https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy">DSPy Integration</a></b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/d81a7c37-382f-4c46-840f-e6c2593301db" alt="MIPROv2 diagram"></td>
    <td width="50%" align="center" valign="middle">
      TensorZero comes with several optimization recipes, but you can also easily create your own.
      This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy, a popular library for automated prompt engineering.
    </td>
  </tr>
</table>

_More coming soon..._

<br>

### 📊 LLM Evaluation

> **Compare prompts, models, and inference strategies using evaluations powered by heuristics and LLM judges.**

- [x] **[Evaluate individual inferences](https://www.tensorzero.com/docs/evaluations/static-evaluations/tutorial)** with _static evaluations_ powered by heuristics or LLM judges (&approx; unit tests for LLMs)
- [x] **[Evaluate end-to-end workflows](https://www.tensorzero.com/docs/evaluations/dynamic-evaluations/tutorial)** with _dynamic evaluations_ with complete flexibility (&approx; integration tests for LLMs)
- [x] Optimize LLM judges just like any other TensorZero function to align them to human preferences
- [ ] Soon: more built-in evaluators; headless evaluations

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b>Evaluation » UI</b></td>
    <td width="50%" align="center" valign="middle"><b>Evaluation » CLI</b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/f4bf54e3-1b63-46c8-be12-2eaabf615699"></td>
    <td width="50%" align="left" valign="middle">
<pre><code class="language-bash">docker compose run --rm evaluations \
  --evaluation-name extract_data \
  --dataset-name hard_test_cases \
  --variant-name gpt_4o \
  --concurrency 5</code></pre>
<pre><code class="language-bash">Run ID: 01961de9-c8a4-7c60-ab8d-15491a9708e4
Number of datapoints: 100
██████████████████████████████████████ 100/100
exact_match: 0.83 ± 0.03
semantic_match: 0.98 ± 0.01
item_count: 7.15 ± 0.39</code></pre>
    </td>
  </tr>
</table>

### 🧪 LLM Experimentation

> **Ship with confidence with built-in A/B testing, routing, fallbacks, retries, etc.**

- [x] Ship with confidence with built-in **[A/B testing](https://www.tensorzero.com/docs/experimentation/run-ab-tests)** for models, prompts, providers, hyperparameters, etc.
- [x] Enforce principled experiments (RCTs) in complex workflows, including multi-turn and compound LLM systems
- [ ] Soon: multi-armed bandits; AI-managed experiments

### & more!

> **Build with an open-source stack well-suited for prototypes but designed from the ground up to support the most complex LLM applications and deployments.**

- [x] Build simple applications or massive deployments with GitOps-friendly orchestration
- [x] **[Extend TensorZero](https://www.tensorzero.com/docs/operations/extend-tensorzero)** with built-in escape hatches, programmatic-first usage, direct database access, and more
- [x] Integrate with third-party tools: specialized observability and evaluations, model providers, agent orchestration frameworks, etc.
- [x] Iterate quickly by experimenting with prompts interactively using the Playground UI

## Frequently Asked Questions

**What is TensorZero?**

TensorZero is an open-source stack for industrial-grade LLM applications. It unifies an LLM gateway, observability, optimization, evaluation, and experimentation.

**How is TensorZero different from other LLM frameworks?**

1. TensorZero enables you to optimize complex LLM applications based on production metrics and human feedback.
2. TensorZero supports the needs of industrial-grade LLM applications: low latency, high throughput, type safety, self-hosted, GitOps, customizability, etc.
3. TensorZero unifies the entire LLMOps stack, creating compounding benefits. For example, LLM evaluations can be used for fine-tuning models alongside AI judges.

**Can I use TensorZero with \_\_\_?**

Yes. Every major programming language is supported. You can use TensorZero with our Python client, any OpenAI SDK or OpenAI-compatible client, or our HTTP API.

**Is TensorZero production-ready?**

Yes. Here's a case study: **[Automating Code Changelogs at a Large Bank with LLMs](https://www.tensorzero.com/blog/case-study-automating-code-changelogs-at-a-large-bank-with-llms)**

**How much does TensorZero cost?**

Nothing. TensorZero is 100% self-hosted and open-source. There are no paid features.

**Who is building TensorZero?**

Our technical team includes a former Rust compiler maintainer, machine learning researchers (Stanford, CMU, Oxford, Columbia) with thousands of citations, and the chief product officer of a decacorn startup. We're backed by the same investors as leading open-source projects (e.g. ClickHouse, CockroachDB) and AI labs (e.g. OpenAI, Anthropic). See our **[$7.3M seed round announcement](https://www.tensorzero.com/blog/tensorzero-raises-7-3m-seed-round-to-build-an-open-source-stack-for-industrial-grade-llm-applications/)** and **[coverage from VentureBeat](https://venturebeat.com/ai/tensorzero-nabs-7-3m-seed-to-solve-the-messy-world-of-enterprise-llm-development/)**. We're **[hiring in NYC](https://www.tensorzero.com/jobs)**.

**How do I get started?**

You can adopt TensorZero incrementally. Our **[Quick Start](https://www.tensorzero.com/docs/quickstart)** goes from a vanilla OpenAI wrapper to a production-ready LLM application with observability and fine-tuning in just 5 minutes.

## Demo

> **Watch LLMs get better at data extraction in real-time with TensorZero!**
>
> **[Dynamic in-context learning (DICL)](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#dynamic-in-context-learning-dicl)** is a powerful inference-time optimization available out of the box with TensorZero.
> It enhances LLM performance by automatically incorporating relevant historical examples into the prompt, without the need for model fine-tuning.

https://github.com/user-attachments/assets/4df1022e-886e-48c2-8f79-6af3cdad79cb

## Get Started

**Start building today.**
The **[Quick Start](https://www.tensorzero.com/docs/quickstart)** shows it's easy to set up an LLM application with TensorZero.

**Questions?**
Ask us on **[Slack](https://www.tensorzero.com/slack)** or **[Discord](https://www.tensorzero.com/discord)**.

**Using TensorZero at work?**
Email us at **[hello@tensorzero.com](mailto:hello@tensorzero.com)** to set up a Slack or Teams channel with your team (free).

## Examples

We are working on a series of **complete runnable examples** illustrating TensorZero's data & learning flywheel.

> **[Optimizing Data Extraction (NER) with TensorZero](https://github.com/tensorzero/tensorzero/tree/main/examples/data-extraction-ner)**
>
> This example shows how to use TensorZero to optimize a data extraction pipeline.
> We demonstrate techniques like fine-tuning and dynamic in-context learning (DICL).
> In the end, an optimized GPT-4o Mini model outperforms GPT-4o on this task &mdash; at a fraction of the cost and latency &mdash; using a small amount of training data.

> **[Agentic RAG — Multi-Hop Question Answering with LLMs](https://github.com/tensorzero/tensorzero/tree/main/examples/rag-retrieval-augmented-generation/simple-agentic-rag/)**
>
> This example shows how to build a multi-hop retrieval agent using TensorZero.
> The agent iteratively searches Wikipedia to gather information, and decides when it has enough context to answer a complex question.

> **[Writing Haikus to Satisfy a Judge with Hidden Preferences](https://github.com/tensorzero/tensorzero/tree/main/examples/haiku-hidden-preferences)**
>
> This example fine-tunes GPT-4o Mini to generate haikus tailored to a specific taste.
> You'll see TensorZero's "data flywheel in a box" in action: better variants leads to better data, and better data leads to better variants.
> You'll see progress by fine-tuning the LLM multiple times.

> **[Image Data Extraction — Multimodal (Vision) Fine-tuning](https://github.com/tensorzero/tensorzero/tree/main/examples/multimodal-vision-finetuning)**
>
> This example shows how to fine-tune multimodal models (VLMs) like GPT-4o to improve their performance on vision-language tasks.
> Specifically, we'll build a system that categorizes document images (screenshots of computer science research papers).

> **[Improving LLM Chess Ability with Best-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles/)**
>
> This example showcases how best-of-N sampling can significantly enhance an LLM's chess-playing abilities by selecting the most promising moves from multiple generated options.

> **[Improving Math Reasoning with a Custom Recipe for Automated Prompt Engineering (DSPy)](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)**
>
> TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows.
> But you can also easily create your own recipes and workflows!
> This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy.

_& many more on the way!_
