import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { TagsEditor } from "./TagsEditor";

const meta: Meta<typeof TagsEditor> = {
  title: "Tags/TagsEditor",
  component: TagsEditor,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof meta>;

// Interactive wrapper component to handle state
const TagsEditorWrapper = ({
  initialTags,
  isEditing,
}: {
  initialTags: Record<string, string>;
  isEditing: boolean;
}) => {
  const [tags, setTags] = useState(initialTags);

  return (
    <div className="w-[600px] p-4">
      <h3 className="text-lg font-semibold mb-4">
        Tags Editor ({isEditing ? "Editing" : "Read-only"})
      </h3>
      <TagsEditor tags={tags} onTagsChange={setTags} isEditing={isEditing} />
    </div>
  );
};

export const ReadOnlyWithTags: Story = {
  render: () => (
    <TagsEditorWrapper
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
  render: () => <TagsEditorWrapper initialTags={{}} isEditing={false} />,
};

export const ReadOnlyWithNavigableSystemTags: Story = {
  render: () => (
    <TagsEditorWrapper
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
    <TagsEditorWrapper
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
  render: () => <TagsEditorWrapper initialTags={{}} isEditing={true} />,
};