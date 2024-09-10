<img src="https://github.com/user-attachments/assets/47d67430-386d-4675-82ad-d4734d3262d9" width=128 height=128>

# TensorZero

**TensorZero builds open-source infrastructure for production-grade, scalable, and complex LLM systems.**

**Why use TensorZero?** It enables a **data & learning flywheel for LLM systems** by integrating inference, observability, optimization, and experimentation.

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
  <b><a href="https://www.tensorzero.com/docs/gateway/tutorial" target="_blank">Tutorial</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Deployment Guide</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/api-reference" target="_blank">API Reference</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/gateway/deployment" target="_blank">Configuration Reference</a></b>
  ·
  <b><a href="https://www.tensorzero.com/docs/vision-roadmap" target="_blank">Vision & Roadmap</a></b>
</p>

## Overview

<br>
<p align="center" >
  <a href="https://www.tensorzero.com/docs">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/34a92c18-242e-4d76-a99c-861283de68a6">
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/e8bc699b-6378-4c2a-9cc1-6d189025e270">
      <img alt="TensorZero Flywheel" src="https://github.com/user-attachments/assets/34a92c18-242e-4d76-a99c-861283de68a6" width=80%>
    </picture>
  </a>
</p>
<br>

1. The **[TensorZero Gateway](https://www.tensorzero.com/docs/gateway/)** is a high-performance model gateway written in Rust 🦀 that provides a unified interface for all your LLM applications.
2. It handles structured schema-based inference with &lt;1ms P99 latency overhead (see **[Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks)**) and built-in observability and experimentation (and soon, inference-time optimizations).
3. It also collects downstream metrics and feedback associated with these inferences, with first-class support for multi-step LLM systems.
4. Everything is stored in a ClickHouse data warehouse that you control for real-time, scalable, and developer-friendly analytics.
5. Over time, **[TensorZero Recipes](https://www.tensorzero.com/docs/recipes)** leverage this structured dataset to optimize your prompts and models: run pre-built recipes for common workflows like fine-tuning, or create your own with complete flexibility using any language and platform.
6. Finally, the gateway's experimentation features and GitOps orchestration enable you to iterate and deploy with confidence, be it a single LLM or thousands of LLMs.

Our goal is to help engineers build, manage, and optimize the next generation of LLM applications: systems that learn from real-world experience.
Read more about our **[Vision & Roadmap](https://www.tensorzero.com/docs/vision-roadmap/)**.

## Get Started

**Next steps?** The **[Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)** shows it's easy to set up an LLM application with TensorZero. It teaches how to build a simple chatbot, an email copilot, a weather RAG system, and a structured data extraction pipeline.

**Questions?** Join our **[Slack](https://www.tensorzero.com/slack)** or **[Discord](https://www.tensorzero.com/discord)** communities. We monitor them closely for questions, feedback, and more.

**Using TensorZero at work?** Email us at **[hello@tensorzero.com](mailto:hello@tensorzero.com)** to set up a Slack or Teams channel with your team (free).

## Examples

We are working on a series of **complete runnable examples** illustrating TensorZero's data & learning flywheel.

> **[Writing Haikus to Satisfy a Judge with Hidden Preferences](https://github.com/tensorzero/tensorzero/tree/main/examples/haiku-hidden-preferences)**
>
> This example fine-tunes GPT-4o Mini to generate haikus tailored to a specific taste.
> You'll see TensorZero's "data flywheel in a box" in action: better variants leads to better data, and better data leads to better variants.
> You'll see progress by fine-tuning the LLM multiple times.

> **[Fine-Tuning TensorZero JSON Functions for Named Entity Recognition (CoNLL++)](https://github.com/tensorzero/tensorzero/tree/main/examples/ner-fine-tuning-json-functions)**
>
> This example shows that an optimized Llama 3.1 8B model can be trained to outperform GPT-4o on an NER task using a small amount of training data, and served by Fireworks at a fraction of the cost and latency.

> **[Automated Prompt Engineering for Math Reasoning (GSM8K) with a Custom Recipe (DSPy)](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)**
>
> TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows.
> But you can also easily create your own recipes and workflows!
> This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy.

_& many more on the way!_
