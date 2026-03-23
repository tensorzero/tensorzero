import type { Meta, StoryObj } from "@storybook/react-vite";
import { Combobox } from "./Combobox";
import { useArgs } from "storybook/preview-api";
import { Box } from "lucide-react";

function generateItems(count: number): string[] {
  return Array.from({ length: count }, (_, i) => `item-${i + 1}`);
}

const meta: Meta<typeof Combobox> = {
  title: "UI/Combobox",
  component: Combobox,
  argTypes: {
    virtualizeThreshold: {
      control: { type: "number", min: 0, max: 1000 },
      description: "Number of items at which virtualization is enabled",
    },
    disabled: {
      control: "boolean",
      description: "Disable the combobox",
    },
    allowCreation: {
      control: "boolean",
      description: "Allow creating new items",
    },
    loading: {
      control: "boolean",
      description: "Show loading state",
    },
    error: {
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
        "getPrefix",
        "getSuffix",
        "getItemDataAttributes",
      ],
    },
  },
};

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    placeholder: "Select item",
    emptyMessage: "No items found",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(10)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

export const WithIcon: Story = {
  args: {
    placeholder: "Select item",
    emptyMessage: "No items found",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(10)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
          getPrefix={() => <Box size={16} className="text-fg-muted" />}
        />
      </div>
    );
  },
};

/**
 * With 100+ items, virtualization kicks in automatically.
 * Only visible items are rendered in the DOM for better performance.
 */
export const Virtualized: Story = {
  args: {
    placeholder: "Search 500 items...",
    emptyMessage: "No items found",
    virtualizeThreshold: 100,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(500)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Force virtualization even with small lists by setting threshold to 0.
 */
export const ForceVirtualized: Story = {
  args: {
    placeholder: "Virtualized (20 items)",
    emptyMessage: "No items found",
    virtualizeThreshold: 0,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(20)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Disable virtualization entirely by setting threshold to Infinity.
 */
export const NoVirtualization: Story = {
  args: {
    placeholder: "No virtualization (200 items)",
    emptyMessage: "No items found",
    virtualizeThreshold: Infinity,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(200)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Stress test with 1000 items to verify virtualization performance.
 */
export const StressTest: Story = {
  args: {
    placeholder: "Search 1000 items...",
    emptyMessage: "No items found",
    virtualizeThreshold: 100,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(1000)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Keyboard navigation in virtualized mode.
 * Arrow Up/Down, Home/End, PageUp/PageDown, Enter, Escape.
 */
export const KeyboardNavigation: Story = {
  args: {
    placeholder: "Click then use arrow keys...",
    emptyMessage: "No items found",
    virtualizeThreshold: 100,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(500)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
        {selected && (
          <p className="text-fg-secondary mt-4 text-sm">
            Selected: <strong>{selected}</strong>
          </p>
        )}
      </div>
    );
  },
};

/**
 * Filtering behavior with virtualization.
 * Type to filter, highlight resets to first match.
 */
export const FilteringDemo: Story = {
  args: {
    placeholder: "Try typing 'item-5' or 'apple'...",
    emptyMessage: "No items found",
    virtualizeThreshold: 50,
  },
  render: function Render(args) {
    const items = [
      ...generateItems(100),
      "apple",
      "banana",
      "cherry",
      "date",
      "elderberry",
    ];
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={items}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Edge case: transition between virtualized and non-virtualized
 * as user filters items below/above threshold.
 */
export const ThresholdTransition: Story = {
  args: {
    placeholder: "Type to filter below 100...",
    emptyMessage: "No items found",
    virtualizeThreshold: 100,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(150)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

/**
 * Virtualized combobox with creation support.
 */
export const VirtualizedWithCreation: Story = {
  args: {
    placeholder: "Search or create...",
    emptyMessage: "No items found",
    virtualizeThreshold: 100,
    allowCreation: true,
    createHint: "Type to create a new item",
    createHeading: "Create new",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          items={generateItems(500)}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
        {selected && (
          <p className="text-fg-secondary mt-4 text-sm">
            Selected: <strong>{selected}</strong>
          </p>
        )}
      </div>
    );
  },
};
