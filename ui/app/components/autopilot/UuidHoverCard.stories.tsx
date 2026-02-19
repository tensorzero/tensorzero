import type { Meta, StoryObj } from "@storybook/react-vite";
import type { FunctionType, ResolvedObject } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import {
  FunctionItem,
  getHoverCardWidth,
  InfoItem,
  type InferencePreview,
  Timestamp,
  TypeBadgeLink,
  VariantItem,
} from "~/components/entity-sheet/UuidHoverCard";

const MOCK_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
const MOCK_TIMESTAMP = "2026-01-15T14:30:00Z";
const MOCK_TIMESTAMP_OLD = "2026-01-15T13:30:00Z";

interface HoverCardShellProps {
  type: ResolvedObject["type"];
  children: React.ReactNode;
}

function HoverCardShell({ type, children }: HoverCardShellProps) {
  return (
    <div
      className={cn(
        "bg-popover text-popover-foreground rounded-md border p-3 shadow-md",
        getHoverCardWidth(type),
      )}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inference hover card stories
// ---------------------------------------------------------------------------

interface InferenceHoverCardProps {
  functionName: string;
  functionType: FunctionType;
  variantName: string;
  variantType: string | null;
  preview: InferencePreview | null;
  isLoading: boolean;
}

function InferenceHoverCard({
  functionName,
  functionType,
  variantName,
  variantType,
  preview,
  isLoading,
}: InferenceHoverCardProps) {
  const obj: Extract<ResolvedObject, { type: "inference" }> = {
    type: "inference",
    function_name: functionName,
    function_type: functionType,
    variant_name: variantName,
    episode_id: "ep-0000",
  };
  return (
    <HoverCardShell type="inference">
      <div className="flex flex-col gap-4">
        <TypeBadgeLink uuid={MOCK_UUID} obj={obj}>
          Inference
        </TypeBadgeLink>
        <FunctionItem functionName={functionName} functionType={functionType} />
        <VariantItem
          functionName={functionName}
          variantName={variantName}
          variantType={variantType}
        />
        <Timestamp data={preview} isLoading={isLoading} />
      </div>
    </HoverCardShell>
  );
}

// ---------------------------------------------------------------------------
// Episode hover card stories
// ---------------------------------------------------------------------------

interface EpisodeHoverCardProps {
  inferenceCount: number | null;
  isLoading: boolean;
}

function EpisodeHoverCard({
  inferenceCount,
  isLoading,
}: EpisodeHoverCardProps) {
  const obj: Extract<ResolvedObject, { type: "episode" }> = {
    type: "episode",
  };
  return (
    <HoverCardShell type="episode">
      <div className="flex flex-col gap-4">
        <TypeBadgeLink uuid={MOCK_UUID} obj={obj}>
          Episode
        </TypeBadgeLink>
        <InfoItem
          label="Inferences"
          value={inferenceCount !== null ? String(inferenceCount) : null}
          isLoading={isLoading}
        />
      </div>
    </HoverCardShell>
  );
}

// ---------------------------------------------------------------------------
// Datapoint hover card stories
// ---------------------------------------------------------------------------

interface DatapointHoverCardProps {
  datapointType: "chat_datapoint" | "json_datapoint";
  datasetName: string;
  functionName: string;
}

function DatapointHoverCard({
  datapointType,
  datasetName,
  functionName,
}: DatapointHoverCardProps) {
  const obj: Extract<
    ResolvedObject,
    { type: "chat_datapoint" | "json_datapoint" }
  > = {
    type: datapointType,
    dataset_name: datasetName,
    function_name: functionName,
  };
  const typeLabel =
    datapointType === "chat_datapoint" ? "Chat Datapoint" : "JSON Datapoint";
  return (
    <HoverCardShell type={datapointType}>
      <div className="flex flex-col gap-4">
        <TypeBadgeLink uuid={MOCK_UUID} obj={obj}>
          {typeLabel}
        </TypeBadgeLink>
        <InfoItem label="Dataset" value={datasetName} />
        <InfoItem label="Function" value={functionName} />
      </div>
    </HoverCardShell>
  );
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

const meta = {
  title: "Autopilot/UuidHoverCard",
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

// ---------------------------------------------------------------------------
// Inference stories
// ---------------------------------------------------------------------------

export const InferenceDefault: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="extract_entities"
      functionType="json"
      variantName="gpt4o_v1"
      variantType="chat_completion"
      preview={{ timestamp: MOCK_TIMESTAMP }}
      isLoading={false}
    />
  ),
};

export const InferenceChatFunction: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="customer_support"
      functionType="chat"
      variantName="claude_sonnet"
      variantType="chat_completion"
      preview={{ timestamp: MOCK_TIMESTAMP_OLD }}
      isLoading={false}
    />
  ),
};

export const InferenceLongFunctionName: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="my_extremely_long_function_name_that_goes_on_and_on_for_a_while"
      functionType="json"
      variantName="gpt4o_v1"
      variantType="chat_completion"
      preview={{ timestamp: MOCK_TIMESTAMP }}
      isLoading={false}
    />
  ),
};

export const InferenceLongVariantName: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="extract_entities"
      functionType="chat"
      variantName="claude_3_5_sonnet_20241022_with_custom_system_prompt_and_fewshot_examples_v3"
      variantType="chat_completion"
      preview={{ timestamp: MOCK_TIMESTAMP }}
      isLoading={false}
    />
  ),
};

export const InferenceLongFunctionAndVariant: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="multi_step_reasoning_with_tool_use_and_retrieval_augmented_generation"
      functionType="json"
      variantName="claude_3_5_sonnet_20241022_with_custom_system_prompt_and_fewshot_examples_v3"
      variantType="chat_completion"
      preview={{ timestamp: MOCK_TIMESTAMP }}
      isLoading={false}
    />
  ),
};

export const InferenceNoVariantType: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="summarize_text"
      functionType="chat"
      variantName="default"
      variantType={null}
      preview={{ timestamp: MOCK_TIMESTAMP }}
      isLoading={false}
    />
  ),
};

export const InferenceLoading: Story = {
  render: () => (
    <InferenceHoverCard
      functionName="extract_entities"
      functionType="json"
      variantName="gpt4o_v1"
      variantType="chat_completion"
      preview={null}
      isLoading={true}
    />
  ),
};

// ---------------------------------------------------------------------------
// Episode stories
// ---------------------------------------------------------------------------

export const EpisodeDefault: Story = {
  render: () => <EpisodeHoverCard inferenceCount={5} isLoading={false} />,
};

export const EpisodeLargeCount: Story = {
  render: () => <EpisodeHoverCard inferenceCount={12847} isLoading={false} />,
};

export const EpisodeLoading: Story = {
  render: () => <EpisodeHoverCard inferenceCount={null} isLoading={true} />,
};

// ---------------------------------------------------------------------------
// Datapoint stories
// ---------------------------------------------------------------------------

export const ChatDatapoint: Story = {
  render: () => (
    <DatapointHoverCard
      datapointType="chat_datapoint"
      datasetName="eval_v2"
      functionName="extract_entities"
    />
  ),
};

export const JsonDatapoint: Story = {
  render: () => (
    <DatapointHoverCard
      datapointType="json_datapoint"
      datasetName="training_set"
      functionName="classify_intent"
    />
  ),
};

export const DatapointLongDatasetName: Story = {
  render: () => (
    <DatapointHoverCard
      datapointType="chat_datapoint"
      datasetName="customer_support_evaluation_dataset_v3_with_multilingual_examples_2026"
      functionName="extract_entities"
    />
  ),
};

export const DatapointLongFunctionName: Story = {
  render: () => (
    <DatapointHoverCard
      datapointType="json_datapoint"
      datasetName="eval_v2"
      functionName="multi_step_reasoning_with_tool_use_and_retrieval_augmented_generation"
    />
  ),
};

export const DatapointLongBoth: Story = {
  render: () => (
    <DatapointHoverCard
      datapointType="chat_datapoint"
      datasetName="customer_support_evaluation_dataset_v3_with_multilingual_examples_2026"
      functionName="multi_step_reasoning_with_tool_use_and_retrieval_augmented_generation"
    />
  ),
};

// ---------------------------------------------------------------------------
// All types side by side
// ---------------------------------------------------------------------------

export const AllTypes: Story = {
  render: () => (
    <div className="flex flex-wrap items-start gap-4">
      <InferenceHoverCard
        functionName="extract_entities"
        functionType="json"
        variantName="gpt4o_v1"
        variantType="chat_completion"
        preview={{ timestamp: MOCK_TIMESTAMP }}
        isLoading={false}
      />
      <EpisodeHoverCard inferenceCount={5} isLoading={false} />
      <DatapointHoverCard
        datapointType="chat_datapoint"
        datasetName="eval_v2"
        functionName="extract_entities"
      />
      <DatapointHoverCard
        datapointType="json_datapoint"
        datasetName="training_set"
        functionName="classify_intent"
      />
    </div>
  ),
};

export const AllTypesLongNames: Story = {
  render: () => (
    <div className="flex flex-wrap items-start gap-4">
      <InferenceHoverCard
        functionName="multi_step_reasoning_with_tool_use_and_retrieval_augmented_generation"
        functionType="json"
        variantName="claude_3_5_sonnet_20241022_with_custom_system_prompt_and_fewshot_examples_v3"
        variantType="chat_completion"
        preview={{ timestamp: MOCK_TIMESTAMP }}
        isLoading={false}
      />
      <EpisodeHoverCard inferenceCount={12847} isLoading={false} />
      <DatapointHoverCard
        datapointType="chat_datapoint"
        datasetName="customer_support_evaluation_dataset_v3_with_multilingual_examples_2026"
        functionName="multi_step_reasoning_with_tool_use_and_retrieval_augmented_generation"
      />
    </div>
  ),
};
