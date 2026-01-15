import type { Meta, StoryObj } from "@storybook/react-vite";
import { ButtonSelect } from "./ButtonSelect";
import { useArgs } from "storybook/preview-api";
import { Table, TablePlus } from "~/components/icons/Icons";

const mockItems = [
  "gpt-4o",
  "gpt-4o-mini",
  "claude-3-5-sonnet",
  "claude-3-5-haiku",
  "gemini-2.0-flash",
  "llama-3.3-70b",
];

const manyItems = Array.from({ length: 100 }, (_, i) => `variant-${i + 1}`);

const meta: Meta<typeof ButtonSelect> = {
  title: "UI/ButtonSelect",
  component: ButtonSelect,
  argTypes: {
    disabled: {
      control: "boolean",
      description: "Disable the select",
    },
    creatable: {
      control: "boolean",
      description: "Allow creating new items",
    },
    isLoading: {
      control: "boolean",
      description: "Show loading state",
    },
    isError: {
      control: "boolean",
      description: "Show error state",
    },
  },
  parameters: {
    controls: {
      exclude: [
        "onSelect",
        "selected",
        "items",
        "trigger",
        "getPrefix",
        "getSuffix",
      ],
    },
  },
};

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={mockItems}
        selected={selected ?? null}
        trigger="Select item"
        placeholder="Search items..."
        emptyMessage="No items found"
        onSelect={(item) => updateArgs({ selected: item })}
      />
    );
  },
};

export const WithSelection: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={mockItems}
        selected={selected ?? "gpt-4o"}
        trigger={selected ?? "gpt-4o"}
        placeholder="Search items..."
        emptyMessage="No items found"
        onSelect={(item) => updateArgs({ selected: item })}
      />
    );
  },
};

export const WithCreation: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={mockItems}
        selected={selected ?? null}
        trigger="Select or create"
        placeholder="Search or create..."
        emptyMessage="No items found"
        creatable
        createHeading="Create new"
        existingHeading="Existing"
        onSelect={(item) => {
          updateArgs({ selected: item });
        }}
      />
    );
  },
};

export const WithPrefixAndSuffix: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={mockItems}
        selected={selected ?? null}
        trigger={
          <span className="flex items-center gap-2">
            <Table size={16} />
            Select model
          </span>
        }
        placeholder="Search models..."
        emptyMessage="No models found"
        getPrefix={(item, isSelected) =>
          isSelected ? (
            <TablePlus size={16} className="text-green-600" />
          ) : (
            <Table size={16} className="text-fg-muted" />
          )
        }
        getSuffix={(item) =>
          item ? <span className="text-fg-tertiary text-xs">model</span> : null
        }
        onSelect={(item) => updateArgs({ selected: item })}
      />
    );
  },
};

export const Loading: Story = {
  render: function Render() {
    return (
      <ButtonSelect
        items={[]}
        selected={null}
        trigger="Select item"
        placeholder="Search items..."
        emptyMessage="No items found"
        isLoading
        loadingMessage="Loading items..."
        onSelect={() => {}}
      />
    );
  },
};

export const Error: Story = {
  render: function Render() {
    return (
      <ButtonSelect
        items={[]}
        selected={null}
        trigger="Select item"
        placeholder="Search items..."
        emptyMessage="No items found"
        isError
        errorMessage="Failed to load items"
        onSelect={() => {}}
      />
    );
  },
};

export const Disabled: Story = {
  render: function Render() {
    return (
      <ButtonSelect
        items={mockItems}
        selected="gpt-4o"
        trigger="gpt-4o"
        placeholder="Search items..."
        emptyMessage="No items found"
        disabled
        onSelect={() => {}}
      />
    );
  },
};

export const ManyItems: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={manyItems}
        selected={selected ?? null}
        trigger="Select variant"
        placeholder="Search variants..."
        emptyMessage="No variants found"
        onSelect={(item) => updateArgs({ selected: item })}
      />
    );
  },
};

export const EmptyItems: Story = {
  render: function Render() {
    return (
      <ButtonSelect
        items={[]}
        selected={null}
        trigger="Select variant"
        placeholder="Search variants..."
        emptyMessage="No variants available"
        onSelect={() => {}}
      />
    );
  },
};

export const NonSearchable: Story = {
  render: function Render() {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <ButtonSelect
        items={mockItems}
        selected={selected ?? null}
        trigger="Select item"
        searchable={false}
        emptyMessage="No items available"
        onSelect={(item) => updateArgs({ selected: item })}
      />
    );
  },
};
