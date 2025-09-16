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
    <div className="w-96 p-4">
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
        user_id: "123",
        experiment: "A",
        "tensorzero::system_tag": "system_value",
        environment: "production",
      }}
      isEditing={false}
    />
  ),
};

export const ReadOnlyEmpty: Story = {
  render: () => <TagsEditorWrapper initialTags={{}} isEditing={false} />,
};

export const EditingMode: Story = {
  render: () => (
    <TagsEditorWrapper
      initialTags={{
        user_id: "123",
        experiment: "A",
        "tensorzero::system_tag": "system_value",
      }}
      isEditing={true}
    />
  ),
};

export const EditingModeEmpty: Story = {
  render: () => <TagsEditorWrapper initialTags={{}} isEditing={true} />,
};