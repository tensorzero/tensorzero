# TensorZero Documentation Migration Plan: Astro/Starlight → Mintlify

## Overview

This document outlines the plan to migrate TensorZero's documentation from Astro/Starlight to Mintlify. The migration involves not just copying content, but also adapting to Mintlify's structure, converting Astro-specific syntax, and reorganizing navigation.

## Current State Analysis

### Mintlify (New) - `/Users/gabrielbianconi/Developer/tensorzero/tensorzero/docs`

**Status**: Partially migrated

- ✅ Basic structure established
- ✅ Some guides migrated (inference, observability, optimization, evaluation, experimentation, deployment, advanced)
- ✅ API reference structure in place
- ✅ Configuration reference placeholder
- ❌ Missing significant content from old docs

### Astro/Starlight (Old) - `/Users/gabrielbianconi/Developer/tensorzero/www/src/content/docs/docs`

**Status**: Comprehensive content that needs migration

- 📝 More detailed gateway documentation
- 📝 Comparison pages with other tools
- 📝 FAQ content
- 📝 More comprehensive evaluations documentation
- 📝 Recipes section
- 📝 Vision & roadmap
- 📝 UI deployment guide

## Migration Strategy

### Phase 1: Content Audit & Mapping

1. **Create comprehensive content inventory**

   - Map old content structure to new Mintlify structure
   - Identify content gaps and overlaps
   - Determine which content needs updating vs. direct migration

2. **Navigation structure reconciliation**
   - Analyze current `docs.json` navigation
   - Plan new navigation structure incorporating missing content
   - Ensure logical content hierarchy

### Phase 2: Content Conversion & Migration

#### Priority 1: Core Missing Content

1. **Gateway Documentation Enhancements**

   - Migrate missing API reference pages
   - Add comprehensive gateway guides (batch inference, episodes, experimentation, etc.)
   - Convert provider-specific documentation
   - Add performance and optimization guides

2. **Evaluations Section Expansion**

   - Migrate dynamic evaluations content
   - Add static evaluations documentation
   - Create proper evaluation index/overview

3. **New Major Sections**
   - Add "Comparisons" section (vs LangChain, LangFuse, LiteLLM, etc.)
   - Migrate "Recipes" section
   - Add FAQ page
   - Add Vision & Roadmap page

#### Priority 2: Content Enhancement & Cleanup

1. **Update existing migrated content**

   - Compare current Mintlify content with Astro versions
   - Update any outdated information
   - Ensure consistency in formatting and style

2. **Asset migration**
   - Copy over images, diagrams, and other assets
   - Update asset references in MDX files
   - Ensure proper image optimization

### Phase 3: Technical Adaptations

#### Syntax Conversions Required

1. **Astro → Mintlify component conversion**

   - `:::tip` → `<Tip>`
   - `:::warning` → `<Warning>`
   - `:::note` → `<Note>`
   - Image imports: Remove Astro `import` statements, use direct paths
   - Update any Astro-specific features

2. **Link updates**

   - Convert internal links to use new structure
   - Update cross-references between pages
   - Ensure all links are relative paths

3. **Frontmatter standardization**
   - Ensure all pages have proper `title` and `description`
   - Remove Astro-specific frontmatter properties
   - Add any Mintlify-specific metadata needed

### Phase 4: Navigation & Structure Updates

0. **Move `guides/inference` to `guides/gateway`**

1. **Update `docs.json`**

   - Add new sections and pages to navigation
   - Reorganize existing structure if needed
   - Ensure proper grouping and hierarchy

2. **Content organization**
   - Ensure logical flow between related pages
   - Add proper cross-references and "next steps" guidance
   - Update any table of contents or index pages

## Detailed Content Migration Checklist

### Missing Major Sections

#### Comparisons

- [ ] `comparison/dspy.mdx`
- [ ] `comparison/langchain.mdx`
- [ ] `comparison/langfuse.mdx`
- [ ] `comparison/litellm.mdx`
- [ ] `comparison/openpipe.mdx`
- [ ] `comparison/openrouter.mdx`
- [ ] `comparison/portkey.mdx`

#### guides/evaluations/features

- [ ] `guides/evaluations/features/index.mdx` (more comprehensive)
- [ ] `guides/evaluations/features/dynamic-evaluations/api-reference.mdx`
- [ ] `guides/evaluations/features/dynamic-evaluations/tutorial.mdx`
- [ ] `guides/evaluations/features/static-evaluations/cli-reference.mdx`
- [ ] `guides/evaluations/features/static-evaluations/configuration-reference.mdx`
- [ ] `guides/evaluations/features/static-evaluations/tutorial.mdx`

#### guides/gateway/features

- [ ] `guides/gateway/features/api-reference/auxiliary-endpoints.mdx`
- [ ] `guides/gateway/features/api-reference/batch-inference.mdx`
- [ ] `guides/gateway/features/api-reference/datasets-datapoints.mdx`
- [ ] `guides/gateway/features/api-reference/feedback.mdx`
- [ ] `guides/gateway/features/api-reference/inference.mdx`
- [ ] `guides/gateway/features/api-reference/inference-openai-compatible.mdx`
- [ ] `guides/gateway/features/guides/batch-inference.mdx`
- [ ] `guides/gateway/features/guides/credential-management.mdx`
- [ ] `guides/gateway/features/guides/episodes.mdx`
- [ ] `guides/gateway/features/guides/experimentation.mdx`
- [ ] `guides/gateway/features/guides/extending-tensorzero.mdx`
- [ ] `guides/gateway/features/guides/inference-caching.mdx`
- [ ] `guides/gateway/features/guides/inference-time-optimizations.mdx`
- [ ] `guides/gateway/features/guides/metrics-feedback.mdx`
- [ ] `guides/gateway/features/guides/multimodal-inference.mdx`
- [ ] `guides/gateway/features/guides/opentelemetry-otlp.mdx`
- [ ] `guides/gateway/features/guides/performance-latency.mdx`
- [ ] `guides/gateway/features/guides/prompt-templates-schemas.mdx`
- [ ] `guides/gateway/features/guides/retries-fallbacks.mdx`
- [ ] `guides/gateway/features/guides/streaming-inference.mdx`
- [ ] `guides/gateway/features/guides/tool-use.mdx`

#### Get started

- [ ] `faq.mdx`
- [ ] `recipes/index.mdx`
- [ ] `vision-roadmap.mdx`
- [ ] `ui/deployment.mdx`

### Content Updates Needed

- [ ] `index.mdx` - Compare and merge enhanced version
- [ ] `quickstart.mdx` - Ensure consistency and completeness
- [ ] Provider documentation - Verify all providers are covered
- [ ] Configuration reference pages

## Technical Considerations

### Asset Handling

- Images need to be copied to new docs directory
- Update all image paths in MDX files
- Ensure proper image optimization for Mintlify

### Component Migration

- Astro components need to be converted to Mintlify equivalents
- Special handling for complex layouts or custom components
- Test all converted components for proper rendering

### Link Integrity & Redirects

- All internal links need to be updated for new structure
- External links should be verified as still valid
- Set up comprehensive redirects for old URLs (see redirect mapping below)

### Redirect Mapping

The following redirects need to be implemented to maintain backward compatibility:

#### Core Documentation Pages
```
/docs -> /
/docs/quickstart -> /quickstart
/docs/faq -> /faq
/docs/recipes -> /recipes
/docs/vision-roadmap -> /vision-roadmap
```

#### Comparison Pages (Stay as is)
```
/docs/comparison/dspy -> /comparison/dspy
/docs/comparison/langchain -> /comparison/langchain
/docs/comparison/langfuse -> /comparison/langfuse
/docs/comparison/litellm -> /comparison/litellm
/docs/comparison/openpipe -> /comparison/openpipe
/docs/comparison/openrouter -> /comparison/openrouter
/docs/comparison/portkey -> /comparison/portkey
```

#### Evaluations → Guides/Evaluation
```
/docs/evaluations -> /guides/evaluation/evaluate-individual-inferences
/docs/evaluations/dynamic-evaluations/api-reference -> /api-reference/introduction
/docs/evaluations/dynamic-evaluations/tutorial -> /guides/evaluation/evaluate-end-to-end-workflows
/docs/evaluations/static-evaluations/cli-reference -> /configuration-reference
/docs/evaluations/static-evaluations/configuration-reference -> /configuration-reference
/docs/evaluations/static-evaluations/tutorial -> /guides/evaluation/evaluate-individual-inferences
```

#### Gateway Core Pages
```
/docs/gateway -> /guides/gateway/call-any-llm-provider
/docs/gateway/index -> /guides/gateway/call-any-llm-provider
/docs/gateway/tutorial -> /guides/advanced/tutorial
/docs/gateway/benchmarks -> /guides/advanced/benchmarks
/docs/gateway/clients -> /guides/advanced/clients
/docs/gateway/configuration-reference -> /configuration-reference
/docs/gateway/data-model -> /guides/advanced/data-model
/docs/gateway/deployment -> /guides/deployment/gateway
/docs/gateway/integrations -> /guides/advanced/integrations
```

#### Gateway API Reference
```
/docs/gateway/api-reference/auxiliary-endpoints -> /api-reference/introduction
/docs/gateway/api-reference/batch-inference -> /api-reference/introduction
/docs/gateway/api-reference/datasets-datapoints -> /api-reference/introduction
/docs/gateway/api-reference/feedback -> /api-reference/introduction
/docs/gateway/api-reference/inference -> /api-reference/introduction
/docs/gateway/api-reference/inference-openai-compatible -> /api-reference/introduction
```

#### Gateway Guides → Observability
```
/docs/gateway/guides/episodes -> /guides/observability/log-episodes
/docs/gateway/guides/metrics-feedback -> /guides/observability/log-feedback
/docs/gateway/guides/opentelemetry-otlp -> /guides/observability/export-otlp-opentelemetry
```

#### Gateway Guides → Gateway
```
/docs/gateway/guides/batch-inference -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/inference-caching -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/multimodal-inference -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/performance-latency -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/prompt-templates-schemas -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/retries-fallbacks -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/streaming-inference -> /guides/gateway/call-any-llm-provider
/docs/gateway/guides/tool-use -> /guides/gateway/call-any-llm-provider
```

#### Gateway Guides → Experimentation & Optimization
```
/docs/gateway/guides/experimentation -> /guides/experimentation/compare-variants-ab-testing
/docs/gateway/guides/inference-time-optimizations -> /guides/optimization/overview
```

#### Gateway Guides → Deployment & Advanced
```
/docs/gateway/guides/credential-management -> /guides/deployment/gateway
/docs/gateway/guides/extending-tensorzero -> /guides/advanced/tutorial
```

#### Model Providers
```
/docs/gateway/guides/providers/anthropic -> /guides/gateway/model-providers/anthropic
/docs/gateway/guides/providers/aws-bedrock -> /guides/gateway/model-providers/aws-bedrock
/docs/gateway/guides/providers/aws-sagemaker -> /guides/gateway/model-providers/aws-sagemaker
/docs/gateway/guides/providers/azure -> /guides/gateway/model-providers/azure
/docs/gateway/guides/providers/deepseek -> /guides/gateway/model-providers/deepseek
/docs/gateway/guides/providers/fireworks -> /guides/gateway/model-providers/fireworks
/docs/gateway/guides/providers/gcp-vertex-ai-anthropic -> /guides/gateway/model-providers/gcp-vertex-ai-anthropic
/docs/gateway/guides/providers/gcp-vertex-ai-gemini -> /guides/gateway/model-providers/gcp-vertex-ai-gemini
/docs/gateway/guides/providers/google-ai-studio-gemini -> /guides/gateway/model-providers/google-ai-studio-gemini
/docs/gateway/guides/providers/groq -> /guides/gateway/model-providers/groq
/docs/gateway/guides/providers/hyperbolic -> /guides/gateway/model-providers/hyperbolic
/docs/gateway/guides/providers/mistral -> /guides/gateway/model-providers/mistral
/docs/gateway/guides/providers/openai -> /guides/gateway/model-providers/openai
/docs/gateway/guides/providers/openai-compatible -> /guides/gateway/model-providers/openai-compatible
/docs/gateway/guides/providers/openrouter -> /guides/gateway/model-providers/openrouter
/docs/gateway/guides/providers/sglang -> /guides/gateway/model-providers/sglang
/docs/gateway/guides/providers/tgi -> /guides/gateway/model-providers/tgi
/docs/gateway/guides/providers/together -> /guides/gateway/model-providers/together
/docs/gateway/guides/providers/vllm -> /guides/gateway/model-providers/vllm
/docs/gateway/guides/providers/xai -> /guides/gateway/model-providers/xai
```

#### UI
```
/docs/ui/deployment -> /guides/deployment/ui
```

**Total: 64 redirects** covering all old documentation URLs to maintain backward compatibility.

## Testing & Validation

1. **Content review**

   - Verify all content renders properly in Mintlify
   - Check for broken links or missing assets
   - Ensure proper formatting and styling

2. **Navigation testing**

   - Test all navigation paths work correctly
   - Verify proper breadcrumbs and cross-references
   - Ensure search functionality works with new content

3. **Cross-platform testing**
   - Test on different devices/screen sizes
   - Verify dark/light mode compatibility
   - Check accessibility compliance

## Timeline Estimation

- **Phase 1** (Content Audit): 1-2 days
- **Phase 2** (Content Migration): 3-5 days
- **Phase 3** (Technical Adaptations): 2-3 days
- **Phase 4** (Navigation & Structure): 1-2 days
- **Testing & Validation**: 1-2 days

**Total Estimated Time**: 8-14 days

## Success Criteria

- [ ] All content from old docs successfully migrated
- [ ] Navigation structure is logical and user-friendly
- [ ] All links and cross-references work correctly
- [ ] Images and assets are properly integrated
- [ ] Content formatting is consistent with Mintlify standards
- [ ] Documentation builds and deploys without errors
- [ ] User testing confirms improved documentation experience

## Next Steps

1. **Review and approve this plan**
2. **Begin Phase 1**: Content audit and detailed mapping
3. **Set up systematic migration process**
4. **Regular checkpoints to ensure quality and consistency**

---

_This plan can be adjusted based on feedback and discoveries during the migration process._
