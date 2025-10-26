import { useState } from "react";
import { TemplateContentBlock } from "./TemplateContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { TemplateInput } from "~/types/tensorzero";

const meta = {
  title: "Input Output/Content Blocks/TemplateContentBlock",
  component: TemplateContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof TemplateContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  name: "Simple",
  args: {
    block: {
      name: "simple",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    },
    isEditing: false,
  },
};

export const SimpleEditing: Story = {
  name: "Simple (Editing)",
  args: {
    block: {
      name: "simple",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SimpleEditingStory() {
    const [block, setBlock] = useState({
      name: "simple",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    } as TemplateInput);
    return (
      <TemplateContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const System: Story = {
  args: {
    block: {
      name: "system",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    },
    isEditing: false,
  },
};

export const SystemEditing: Story = {
  name: "System (Editing)",
  args: {
    block: {
      name: "system",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SystemEditingStory() {
    const [block, setBlock] = useState({
      name: "system",
      arguments: {
        role: "programmer",
        iq: 150,
        tools: {
          programming: ["VSCode", "JetBrains"],
          design: ["Figma", "Sketch"],
          writing: ["Grammarly", "ProWritingAid"],
        },
      },
    } as TemplateInput);
    return (
      <TemplateContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const LongValues: Story = {
  args: {
    block: {
      name: "very_".repeat(1000) + "long_name",
      arguments: {
        long_text:
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        long_array: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      },
    },
    isEditing: false,
  },
};

export const LongValuesEditing: Story = {
  name: "Long Values (Editing)",
  args: {
    block: {
      name: "very_".repeat(1000) + "long_name",
      arguments: {
        long_text:
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        long_array: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      },
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function LongValuesEditingStory() {
    const [block, setBlock] = useState({
      name: "very_".repeat(1000) + "long_name",
      arguments: {
        long_text:
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        long_array: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      },
    } as TemplateInput);
    return (
      <TemplateContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const Empty: Story = {
  args: {
    block: {
      name: "",
      arguments: {},
    },
    isEditing: false,
  },
};
