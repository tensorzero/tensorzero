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

/**
 * Demo keyboard navigation in virtualized mode.
 * - Arrow Up/Down: Move highlight
 * - Home/End: Jump to first/last
 * - PageUp/PageDown: Jump ~8 items
 * - Enter: Select highlighted item
 * - Escape: Close dropdown
 */
export const KeyboardNavigation: Story = {
  args: {
    items: generateItems(500),
    virtualizeThreshold: 100,
    placeholder: "Click then use arrow keys...",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          <strong>Keyboard shortcuts:</strong>
          <br />
          ↑↓ Move highlight
          <br />
          Home/End Jump to start/end
          <br />
          PageUp/PageDown Jump 8 items
          <br />
          Enter Select
          <br />
          Escape Close
        </p>
        <Combobox
          {...args}
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
 * Demo filtering behavior with virtualization.
 * Type to filter, selection persists through filter changes.
 */
export const FilteringDemo: Story = {
  args: {
    items: [
      ...generateItems(100),
      "apple",
      "banana",
      "cherry",
      "date",
      "elderberry",
    ],
    virtualizeThreshold: 50,
    placeholder: "Try typing 'item' or 'apple'...",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          Type to filter. Try &quot;item_5&quot; or &quot;apple&quot;. Highlight
          resets to first match when filtering.
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
 * Edge case: transition between virtualized and non-virtualized
 * as user filters items below/above threshold.
 */
export const ThresholdTransition: Story = {
  args: {
    items: generateItems(150),
    virtualizeThreshold: 100,
    placeholder: "Type to filter below 100...",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <div className="w-80">
        <p className="text-fg-muted mb-4 text-sm">
          150 items with threshold=100. Type to filter below 100 items and
          observe transition from virtualized to non-virtualized rendering.
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
