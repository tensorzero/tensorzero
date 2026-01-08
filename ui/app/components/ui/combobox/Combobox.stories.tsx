import type { Meta, StoryObj } from "@storybook/react-vite";
import { Combobox } from "./Combobox";
import { useArgs } from "storybook/preview-api";
import { Box } from "lucide-react";

function generateItems(count: number): string[] {
  return Array.from({ length: count }, (_, i) => `item_${i + 1}`);
}

const meta = {
  title: "UI/Combobox",
  component: Combobox,
  argTypes: {
    virtualizeThreshold: {
      control: { type: "number", min: 0, max: 1000 },
      description: "Number of items at which virtualization is enabled",
    },
  },
  parameters: {
    controls: {
      exclude: ["onSelect", "selected", "items", "getItemIcon"],
    },
  },
  args: {
    onSelect: () => {},
    placeholder: "Select item",
    emptyMessage: "No items found",
  },
} satisfies Meta<typeof Combobox>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    items: generateItems(10),
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};

export const WithIcon: Story = {
  args: {
    items: generateItems(10),
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <Combobox
          {...args}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
          getItemIcon={() => <Box size={16} className="text-fg-muted" />}
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
    items: generateItems(500),
    virtualizeThreshold: 100,
    placeholder: "Search 500 items...",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          This combobox has 500 items. Virtualization renders only visible items
          for smooth scrolling.
        </p>
        <Combobox
          {...args}
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
    items: generateItems(20),
    virtualizeThreshold: 0,
    placeholder: "Virtualized (20 items)",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          Virtualization forced on with only 20 items (threshold=0).
        </p>
        <Combobox
          {...args}
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
    items: generateItems(200),
    virtualizeThreshold: Infinity,
    placeholder: "No virtualization (200 items)",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          200 items rendered without virtualization (threshold=Infinity). May be
          slower.
        </p>
        <Combobox
          {...args}
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
    items: generateItems(1000),
    virtualizeThreshold: 100,
    placeholder: "Search 1000 items...",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          Stress test: 1000 items with virtualization. Should scroll smoothly.
        </p>
        <Combobox
          {...args}
          selected={selected ?? null}
          onSelect={(item) => updateArgs({ selected: item })}
        />
      </div>
    );
  },
};
