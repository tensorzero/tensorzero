<p><picture><img src="https://github.com/user-attachments/assets/47d67430-386d-4675-82ad-d4734d3262d9" alt="TensorZero Logo" width=128 height=128></picture></p>

# TensorZero

<p><picture><img src="https://www.tensorzero.com/github-trending-badge.svg" alt="#1 Repository Of The Day"></picture></p>

**TensorZero is an open-source stack for _industrial-grade LLM applications_:**

- [x] **Gateway**
  - [x] Access every major LLM provider (API or self-hosted) through a single unified API
  - [x] Infer with streaming, tool use, structured generation (JSON mode), batch, multimodal (VLMs), file inputs, caching, etc.
  - [x] Define prompt templates and schemas to enforce a consistent, typed interface between your application and the LLMs
  - [x] Satisfy extreme throughput and latency needs, thanks to Rust: <1ms p99 latency overhead at 10k+ QPS
  - [x] Integrate using our Python client, any OpenAI SDK or OpenAI-compatible client, or our HTTP API (use any programming language)
  - [x] Ensure high availability with routing, retries, fallbacks, load balancing, granular timeouts, etc.
  - [ ] Soon: embeddings; real-time voice  
- [x] **Observability**
  - [x] Store inferences and feedback (metrics, human edits, etc.) in your own database
  - [x] Dive into individual inferences or high-level aggregate patterns using the TensorZero UI or programmatically
  - [x] Build datasets for optimization, evaluations, and other workflows
  - [x] Replay historical inferences with new prompts, models, inference strategies, etc.
  - [x] Export OpenTelemetry (OTLP) traces to your favorite general-purpose observability tool
  - [ ] Soon: AI-assisted debugging and root cause analysis; AI-assisted data labeling
- [x] **Optimization**
  - [x] Optimize your models with supervised fine-tuning, RLHF, and other techniques
  - [x] Optimize your prompts with automated prompt engineering algorithms like MIPROv2
  - [x] Optimize your inference strategy with dynamic in-context learning, chain of thought, best/mixture-of-N sampling, etc.
  - [x] Enable a feedback loop for your LLMs: a data & learning flywheel turning production data into smarter, faster, and cheaper models
  - [ ] Soon: programmatic optimization; synthetic data generation
- [x] **Evaluations**
  - [x] Evaluate individual inferences with _static evaluations_ powered by heuristics or LLM judges (&approx; unit tests for LLMs)
  - [x] Evaluate end-to-end workflows with _dynamic evaluations_ with complete flexibility (&approx; integration tests for LLMs)
  - [x] Optimize LLM judges just like any other TensorZero function to align them to human preferences
  - [ ] Soon: more built-in evaluators; headless evaluations
- [x] **Experimentation**
  - [x] Ship with confidence with built-in A/B testing for models, prompts, providers, hyperparameters, etc.
  - [x] Enforce principled experiments (RCTs) in complex workflows, including multi-turn and compound LLM systems
  - [ ] Soon: multi-armed bandits; AI-managed experiments
- [x] **& more!**
  - [x] Build simple applications or massive deployments with GitOps-friendly orchestration
  - [x] Extend TensorZero with built-in escape hatches, programmatic-first usage, direct database access, and more
  - [x] Integrate with third-party tools: specialized observability and evaluations, model providers, agent orchestration frameworks, etc.
  - [ ] Soon: UI playground

Take what you need, adopt incrementally, and complement with other tools.

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
  <b><a href="https://www.tensorzero.com/docs/gateway/tutorial" target="_blank">Comprehensive Tutorial</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Deployment Guide</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/api-reference" target="_blank">API Reference</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Configuration Reference</a></b>
</p>

---

<table>
  <tr>
    <td width="30%" valign="top"><b>What is TensorZero?</b></td>
    <td width="70%" valign="top">TensorZero is an open-source stack for industrial-grade LLM applications. It unifies an LLM gateway, observability, optimization, evaluations, and experimentation.</td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>How is TensorZero different from other LLM frameworks?</b></td>
    <td width="70%" valign="top">
      1. TensorZero enables you to optimize complex LLM applications based on production metrics and human feedback.<br>
      2. TensorZero supports the needs of industrial-grade LLM applications: low latency, high throughput, type safety, self-hosted, GitOps, customizability, etc.<br>
      3. TensorZero unifies the entire LLMOps stack, creating compounding benefits. For example, LLM evaluations can be used for fine-tuning models alongside AI judges.
    </td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>Can I use TensorZero with ___?</b></td>
    <td width="70%" valign="top">Yes. Every major programming language is supported. You can use TensorZero with our Python client, any OpenAI SDK or OpenAI-compatible client, or our HTTP API.</td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>Is TensorZero production-ready?</b></td>
    <td width="70%" valign="top">Yes. Here's a case study: <b><a href="https://www.tensorzero.com/blog/case-study-automating-code-changelogs-at-a-large-bank-with-llms">Automating Code Changelogs at a Large Bank with LLMs</a></b></td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>How much does TensorZero cost?</b></td>
    <td width="70%" valign="top">Nothing. TensorZero is 100% self-hosted and open-source. There are no paid features.</td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>Who is building TensorZero?</b></td>
    <td width="70%" valign="top">Our technical team includes a former Rust compiler maintainer, machine learning researchers (Stanford, CMU, Oxford, Columbia) with thousands of citations, and the chief product officer of a decacorn startup. We're backed by the same investors as leading open-source projects (e.g. ClickHouse, CockroachDB) and AI labs (e.g. OpenAI, Anthropic).</td>
  </tr>
  <tr>
    <td width="30%" valign="top"><b>How do I get started?</b></td>
    <td width="70%" valign="top">You can adopt TensorZero incrementally. Our <b><a href="https://www.tensorzero.com/docs/quickstart">Quick Start</a></b> goes from a vanilla OpenAI wrapper to a production-ready LLM application with observability and fine-tuning in just 5 minutes.</td>
  </tr>
</table>

---

## Features

### 🌐 LLM Gateway

> **Integrate with TensorZero once and access every major LLM provider.**

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b>Model Providers</b></td>
    <td width="50%" align="center" valign="middle"><b>Features</b></td>
  </tr>
  <tr>
    <td width="50%" align="left" valign="top">
      <p>
        The TensorZero Gateway natively supports:
      </p>
      <ul>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/anthropic">Anthropic</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/aws-bedrock">AWS Bedrock</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/aws-sagemaker">AWS SageMaker</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/azure">Azure OpenAI Service</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/deepseek">DeepSeek</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/fireworks">Fireworks</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-anthropic">GCP Vertex AI Anthropic</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-gemini">GCP Vertex AI Gemini</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/google-ai-studio-gemini">Google AI Studio (Gemini API)</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/hyperbolic">Hyperbolic</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/mistral">Mistral</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/openai">OpenAI</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/together">Together</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/vllm">vLLM</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/xai">xAI</a></b></li>
      </ul>
      <p>
        <em>
          Need something else?
          Your provider is most likely supported because TensorZero integrates with <b><a href="https://www.tensorzero.com/docs/gateway/guides/providers/openai-compatible">any OpenAI-compatible API (e.g. Ollama)</a></b>.
          </em>
      </p>
    </td>
    <td width="50%" align="left" valign="top">
      <p>
        The TensorZero Gateway supports advanced features like:
      </p>
      <ul>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks">Retries & Fallbacks</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations">Inference-Time Optimizations</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/prompt-templates-schemas">Prompt Templates & Schemas</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/tutorial#experimentation">Experimentation (A/B Testing)</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/configuration-reference">Configuration-as-Code (GitOps)</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/batch-inference">Batch Inference</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/multimodal-inference">Multimodal Inference (VLMs)</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/inference-caching">Inference Caching</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/metrics-feedback">Metrics & Feedback</a></b></li>
        <li><b><a href="https://www.tensorzero.com/docs/gateway/guides/episodes">Multi-Step LLM Workflows (Episodes)</a></b></li>
        <li><em>& a lot more...</em></li>
      </ul>
      <p>
        The TensorZero Gateway is written in Rust 🦀 with <b>performance</b> in mind (&lt;1ms p99 latency overhead @ 10k QPS).
        See <b><a href="https://www.tensorzero.com/docs/gateway/benchmarks">Benchmarks</a></b>.<br>
      </p>
      <p>
        You can run inference using the <b>TensorZero client</b> (recommended), the <b>OpenAI client</b>, or the <b>HTTP API</b>.
      </p>
    </td>
  </tr>
</table>

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
<summary><b>Usage: Python &mdash; OpenAI Client</b></summary>

You can access any provider using the OpenAI Python client with TensorZero.

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
<summary><b>Usage: JavaScript / TypeScript (Node) &mdash; OpenAI Client</b></summary>

You can access any provider using the OpenAI Node client with TensorZero.

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

### 📈 LLM Optimization

> **Send production metrics and human feedback to easily optimize your prompts, models, and inference strategies &mdash; using the UI or programmatically.**

#### Model Optimization

Optimize closed-source and open-source models using supervised fine-tuning (SFT) and preference fine-tuning (DPO).

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b>Supervised Fine-tuning &mdash; UI</b></td>
    <td width="50%" align="center" valign="middle"><b>Preference Fine-tuning (DPO) &mdash; Jupyter Notebook</b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/cf7acf66-732b-43b3-af2a-5eba1ce40f6f"></td>
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

### 🔍 LLM Observability

> **Zoom in to debug individual API calls, or zoom out to monitor metrics across models and prompts over time &mdash; all using the open-source TensorZero UI.**

<table>
  <tr></tr> <!-- flip highlight order -->
  <tr>
    <td width="50%" align="center" valign="middle"><b>Observability » Inference</b></td>
    <td width="50%" align="center" valign="middle"><b>Observability » Function</b></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/2cc3cc9a-f33f-4e94-b8de-07522326f80a"></td>
    <td width="50%" align="center" valign="middle"><img src="https://github.com/user-attachments/assets/00ae6605-8fa0-4efd-8238-ae8ea589860f"></td>
  </tr>
</table>

<br>

### 📊 LLM Evaluations

> **Compare prompts, models, and inference strategies using TensorZero Evaluations &mdash; with support for heuristics and LLM judges.**

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

## Demo

> **Watch LLMs get better at data extraction in real-time with TensorZero!**
>
> **[Dynamic in-context learning (DICL)](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#dynamic-in-context-learning-dicl)** is a powerful inference-time optimization available out of the box with TensorZero.
> It enhances LLM performance by automatically incorporating relevant historical examples into the prompt, without the need for model fine-tuning.

https://github.com/user-attachments/assets/4df1022e-886e-48c2-8f79-6af3cdad79cb

## Get Started

**Start building today.**
The **[Quick Start](https://www.tensorzero.com/docs/quickstart)** shows it's easy to set up an LLM application with TensorZero.
If you want to dive deeper, the **[Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)** teaches how to build a simple chatbot, an email copilot, a weather RAG system, and a structured data extraction pipeline.

**Questions?**
Ask us on **[Slack](https://www.tensorzero.com/slack)** or **[Discord](https://www.tensorzero.com/discord)**.

**Using TensorZero at work?**
Email us at **[hello@tensorzero.com](mailto:hello@tensorzero.com)** to set up a Slack or Teams channel with your team (free).

**Work with us.**
We're **[hiring in NYC](https://www.tensorzero.com/jobs)**.
We'd also welcome **[open-source contributions](https://github.com/tensorzero/tensorzero/blob/main/CONTRIBUTING.md)**!

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

> **[Improving LLM Chess Ability with Best-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles/)**
>
> This example showcases how best-of-N sampling can significantly enhance an LLM's chess-playing abilities by selecting the most promising moves from multiple generated options.

> **[Improving Math Reasoning with a Custom Recipe for Automated Prompt Engineering (DSPy)](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)**
>
> TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows.
> But you can also easily create your own recipes and workflows!
> This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy.

_& many more on the way!_
