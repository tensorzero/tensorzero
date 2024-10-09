<img src="https://github.com/user-attachments/assets/47d67430-386d-4675-82ad-d4734d3262d9" width=128 height=128>

# TensorZero

**TensorZero enables LLM applications that learn from real-world experience.**

1. Integrate our model gateway
2. Send metrics or feedback
3. Optimize prompts, models, and inference-time strategies
4. Unlock compounding improvements in quality, cost, and latency

It provides a **data & learning flywheel for LLMs** by unifying:

- [x] **Inference:** one API for all LLMs, with <1ms P99 overhead
- [x] **Observability:** inference & feedback → your database
- [x] **Optimization:** from prompts to fine-tuning and RL (& even 🍓? **[→](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)**) 
- [x] **Experimentation:** built-in A/B testing, routing, fallbacks

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
  <b><a href="https://www.tensorzero.com/docs/gateway/quickstart" target="_blank">Quick Start (5min)</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/tutorial" target="_blank">Comprehensive Tutorial</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Deployment Guide</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/api-reference" target="_blank">API Reference</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Configuration Reference</a></b>
</p>

## Overview

<br>
<p align="center" >
  <a href="https://www.tensorzero.com/docs">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/34a92c18-242e-4d76-a99c-861283de68a6">
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/e8bc699b-6378-4c2a-9cc1-6d189025e270">
      <img alt="TensorZero Flywheel" src="https://github.com/user-attachments/assets/34a92c18-242e-4d76-a99c-861283de68a6" width=720>
    </picture>
  </a>
</p>
<br>

1. The **[TensorZero Gateway](https://www.tensorzero.com/docs/gateway/)** is a high-performance model gateway written in Rust 🦀 that provides a unified API interface for all major LLM providers, allowing for seamless cross-platform integration and fallbacks.
2. It handles structured schema-based inference with &lt;1ms P99 latency overhead (see **[Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks)**) and built-in observability, experimentation, and **[inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)**.
3. It also collects downstream metrics and feedback associated with these inferences, with first-class support for multi-step LLM systems.
4. Everything is stored in a ClickHouse data warehouse that you control for real-time, scalable, and developer-friendly analytics.
5. Over time, **[TensorZero Recipes](https://www.tensorzero.com/docs/recipes)** leverage this structured dataset to optimize your prompts and models: run pre-built recipes for common workflows like fine-tuning, or create your own with complete flexibility using any language and platform.
6. Finally, the gateway's experimentation features and GitOps orchestration enable you to iterate and deploy with confidence, be it a single LLM or thousands of LLMs.

Our goal is to help engineers build, manage, and optimize the next generation of LLM applications: systems that learn from real-world experience.
Read more about our **[Vision & Roadmap](https://www.tensorzero.com/docs/vision-roadmap/)**.

## Get Started

**Next steps?** The [Quick Start](https://www.tensorzero.com/docs/gateway/quickstart) shows it's easy to set up an LLM application with TensorZero. If you want to dive deeper, the [Tutorial](https://www.tensorzero.com/docs/gateway/tutorial) teaches how to build a simple chatbot, an email copilot, a weather RAG system, and a structured data extraction pipeline.

**Questions?** Ask us on **[Slack](https://www.tensorzero.com/slack)** or **[Discord](https://www.tensorzero.com/discord)**.

**Using TensorZero at work?** Email us at **[hello@tensorzero.com](mailto:hello@tensorzero.com)** to set up a Slack or Teams channel with your team (free).

## Examples

We are working on a series of **complete runnable examples** illustrating TensorZero's data & learning flywheel.

> **[Writing Haikus to Satisfy a Judge with Hidden Preferences](https://github.com/tensorzero/tensorzero/tree/main/examples/haiku-hidden-preferences)**
>
> This example fine-tunes GPT-4o Mini to generate haikus tailored to a specific taste.
> You'll see TensorZero's "data flywheel in a box" in action: better variants leads to better data, and better data leads to better variants.
> You'll see progress by fine-tuning the LLM multiple times.

> **[Improving Data Extraction (NER) by Fine-Tuning a Llama 3 Model](https://github.com/tensorzero/tensorzero/tree/main/examples/ner-fine-tuning)**
>
> This example shows that an optimized Llama 3.1 8B model can be trained to outperform GPT-4o on a Named Entity Recognition (NER) task using a small amount of training data, and served by Fireworks at a fraction of the cost and latency.

> **[Improving LLM Chess Ability with Best-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles-best-of-n-sampling/)**
>
> This example showcases how best-of-N sampling can significantly enhance an LLM's chess-playing abilities by selecting the most promising moves from multiple generated options.

> **[Improving Data Extraction (NER) with Dynamic In-Context Learning](https://github.com/tensorzero/tensorzero/tree/main/examples/ner-dicl)**
>
> This example demonstrates how Dynamic In-Context Learning (DICL) can enhance Named Entity Recognition (NER) performance by leveraging relevant historical examples to improve data extraction accuracy and consistency without having to fine-tune a model.

> **[Improving Math Reasoning with a Custom Recipe for Automated Prompt Engineering (DSPy)](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)**
>
> TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows.
> But you can also easily create your own recipes and workflows!
> This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy.

_& many more on the way!_
