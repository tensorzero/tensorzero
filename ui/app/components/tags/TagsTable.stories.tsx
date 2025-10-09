import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { TagsTable } from "./TagsTable";

const meta: Meta<typeof TagsTable> = {
  title: "Tags/TagsTable",
  component: TagsTable,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof meta>;

// Interactive wrapper component to handle state
const TagsTableWrapper = ({
  initialTags,
  isEditing,
}: {
  initialTags: Record<string, string>;
  isEditing: boolean;
}) => {
  const [tags, setTags] = useState(initialTags);

  return (
    <div className="w-[600px] p-4">
      <h3 className="mb-4 text-lg font-semibold">
        Tags Table ({isEditing ? "Editing" : "Read-only"})
      </h3>
      <TagsTable tags={tags} onTagsChange={setTags} isEditing={isEditing} />
    </div>
  );
};

export const ReadOnlyWithTags: Story = {
  render: () => (
    <TagsTableWrapper
      initialTags={{
        apple: "fruit",
        banana: "yellow",
        "tensorzero::system_tag": "system_value",
        zebra: "animal",
      }}
      isEditing={false}
    />
  ),
};

export const ReadOnlyEmpty: Story = {
  render: () => <TagsTableWrapper initialTags={{}} isEditing={false} />,
};

export const ReadOnlyWithNavigableSystemTags: Story = {
  render: () => (
    <TagsTableWrapper
      initialTags={{
        "tensorzero::dataset_name": "sample_dataset",
        "tensorzero::datapoint_id": "123456789",
        "tensorzero::evaluation_name": "test_evaluation",
        "tensorzero::evaluation_run_id": "run_abc123",
        "tensorzero::evaluator_inference_id": "inference_xyz789",
        user_tag: "custom_value",
      }}
      isEditing={false}
    />
  ),
};

export const EditingMode: Story = {
  render: () => (
    <TagsTableWrapper
      initialTags={{
        apple: "fruit",
        banana: "yellow",
        "tensorzero::system_tag": "system_value",
        zebra: "animal",
      }}
      isEditing={true}
    />
  ),
};

export const EditingModeEmpty: Story = {
  render: () => <TagsTableWrapper initialTags={{}} isEditing={true} />,
};
