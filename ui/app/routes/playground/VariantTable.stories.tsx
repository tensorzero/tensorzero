import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { VariantTable } from "./VariantTable";
import { type VariantConfig } from "~/utils/config/variant";

const meta: Meta<typeof VariantTable> = {
  title: "Components/VariantTable",
  component: VariantTable,
  parameters: {
    layout: "padded",
  },
  argTypes: {
    onVariantSelect: { action: "variant selected" },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Mock variant data showcasing all different types
const mockVariants: [string, VariantConfig][] = [
  [
    "primary-gpt4",
    {
      type: "chat_completion",
      weight: 0.8,
      model: "gpt-4o",
      temperature: 0.7,
      max_tokens: 2000,
      top_p: 0.9,
      presence_penalty: 0.1,
      frequency_penalty: 0.1,
      json_mode: "on",
      retries: { num_retries: 3, max_delay_s: 10 },
    },
  ],
  [
    "fallback-claude",
    {
      type: "chat_completion", 
      weight: 0.2,
      model: "claude-3-5-sonnet-20241022",
      temperature: 0.5,
      max_tokens: 1500,
      json_mode: "strict",
      retries: { num_retries: 2, max_delay_s: 5 },
    },
  ],
  [
    "best-of-3-ensemble",
    {
      type: "experimental_best_of_n_sampling",
      weight: 0.3,
      timeout_s: 300,
      candidates: ["primary-gpt4", "fallback-claude", "backup-model"],
      evaluator: {
        model: "gpt-4o-mini",
        temperature: 0.1,
        max_tokens: 100,
        json_mode: "on",
        retries: { num_retries: 1, max_delay_s: 5 },
      },
    },
  ],
  [
    "rag-dicl",
    {
      type: "experimental_dynamic_in_context_learning",
      weight: 0.4,
      embedding_model: "text-embedding-3-large",
      k: 5,
      model: "gpt-4o",
      temperature: 0.3,
      max_tokens: 3000,
      json_mode: "on",
      retries: { num_retries: 2, max_delay_s: 8 },
    },
  ],
  [
    "mixture-ensemble",
    {
      type: "experimental_mixture_of_n",
      weight: 0.6,
      timeout_s: 450,
      candidates: ["primary-gpt4", "fallback-claude"],
      fuser: {
        weight: 0,
        model: "gpt-4o",
        temperature: 0.2,
        max_tokens: 2500,
        json_mode: "strict",
        retries: { num_retries: 1, max_delay_s: 5 },
      },
    },
  ],
  [
    "chain-of-thought",
    {
      type: "experimental_chain_of_thought",
      weight: 0.5,
      model: "gpt-4o",
      temperature: 0.8,
      max_tokens: 4000,
      top_p: 0.95,
      json_mode: "off",
      retries: { num_retries: 3, max_delay_s: 15 },
    },
  ],
];

const smallVariantSet: [string, VariantConfig][] = [
  [
    "simple-gpt4",
    {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o",
      temperature: 0.7,
      json_mode: "on",
      retries: { num_retries: 0, max_delay_s: 10 },
    },
  ],
  [
    "simple-claude",
    {
      type: "chat_completion",
      weight: 0.5,
      model: "claude-3-5-sonnet-20241022",
      temperature: 0.5,
      json_mode: "strict",
      retries: { num_retries: 0, max_delay_s: 10 },
    },
  ],
];

// Interactive story with selection state
const VariantTableWithSelection = (args: any) => {
  const [selectedVariant, setSelectedVariant] = useState<string | undefined>();

  return (
    <div className="space-y-4">
      <div className="p-4 bg-muted rounded-lg">
        <p className="text-sm font-medium">
          Selected Variant: {selectedVariant || "None"}
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          Left-click column headers for options • Drag column edges to resize • Smart column hiding
        </p>
      </div>
      <VariantTable
        {...args}
        selectedVariant={selectedVariant}
        onVariantSelect={setSelectedVariant}
      />
    </div>
  );
};

export const Default: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: mockVariants,
  },
};

export const AllVariantTypes: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: mockVariants,
  },
  parameters: {
    docs: {
      description: {
        story: "Showcases all available variant types: Chat Completion, Best of N Sampling, Dynamic In-Context Learning, Mixture of N, and Chain of Thought.",
      },
    },
  },
};

export const SimpleConfiguration: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: smallVariantSet,
  },
  parameters: {
    docs: {
      description: {
        story: "A simpler configuration with just two basic chat completion variants.",
      },
    },
  },
};

export const SingleVariant: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: [mockVariants[0]],
  },
  parameters: {
    docs: {
      description: {
        story: "Table with a single variant to test minimal configuration.",
      },
    },
  },
};

export const EmptyState: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: [],
  },
  parameters: {
    docs: {
      description: {
        story: "Empty state when no variants are available.",
      },
    },
  },
};

export const ExperimentalVariants: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: mockVariants.filter(([_, config]) => 
      config.type.startsWith("experimental_")
    ),
  },
  parameters: {
    docs: {
      description: {
        story: "Showcases only experimental variant types: Best of N, DICL, Mixture of N, and Chain of Thought.",
      },
    },
  },
};

export const SmartColumnDefaults: Story = {
  render: VariantTableWithSelection,
  args: {
    variants: [
      [
        "identical-temp-1",
        {
          type: "chat_completion",
          weight: 0.5,
          model: "gpt-4o",
          temperature: 0.7,
          json_mode: "on",
          retries: { num_retries: 0, max_delay_s: 10 },
        },
      ],
      [
        "identical-temp-2", 
        {
          type: "chat_completion",
          weight: 0.5,
          model: "claude-3-5-sonnet-20241022",
          temperature: 0.7, // Same temperature - should be hidden
          json_mode: "on", // Same json_mode - should be hidden
          retries: { num_retries: 0, max_delay_s: 10 },
        },
      ],
    ],
  },
  parameters: {
    docs: {
      description: {
        story: "Demonstrates smart column defaults - Temperature and JSON Mode columns are hidden because all variants have the same values. Model column is shown because values differ.",
      },
    },
  },
};

// Story demonstrating column interaction features
export const InteractionGuide: Story = {
  render: (args: any) => (
    <div className="space-y-6">
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">Table Interaction Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900">Row Selection</h4>
            <p className="text-blue-700 mt-1">Click any row to select a variant</p>
          </div>
          <div className="p-3 bg-green-50 rounded-lg">
            <h4 className="font-medium text-green-900">Column Management</h4>
            <p className="text-green-700 mt-1">Left-click headers → "Show Columns" submenu</p>
          </div>
          <div className="p-3 bg-purple-50 rounded-lg">
            <h4 className="font-medium text-purple-900">Smart Defaults</h4>
            <p className="text-purple-700 mt-1">Only columns with varying data shown by default</p>
          </div>
          <div className="p-3 bg-orange-50 rounded-lg">
            <h4 className="font-medium text-orange-900">Sorting</h4>
            <p className="text-orange-700 mt-1">Use column context menu for sorting options</p>
          </div>
        </div>
      </div>
      <VariantTableWithSelection {...args} />
    </div>
  ),
  args: {
    variants: mockVariants,
  },
  parameters: {
    docs: {
      description: {
        story: "Interactive guide showing all the table features including row selection, column management, and context menus.",
      },
    },
  },
};