import type { Meta, StoryObj } from "@storybook/react-vite";
import { DatasetSelector } from "./DatasetSelector";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useArgs } from "storybook/preview-api";

// Ordered incorrectly to test sorting
const mockDatasets = [
  {
    name: "test_dataset",
    count: 3250,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
  },
  {
    name: "evaluation_set",
    count: 850,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
  },
  {
    name: "training_data_v2",
    count: 42100,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(), // 3 days ago
  },
  {
    name: "validation_set",
    count: 1200,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week ago
  },
  {
    name: "production_data",
    count: 15420,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 5).toISOString(), // 5 minutes ago
  },
  {
    name: "long_dataset_name_that_should_still_be_displayed_gracefully_somehow",
    count: 1,
    lastUpdated: new Date(Date.now()).toISOString(), // now
  },
];

const meta = {
  title: "Dataset/DatasetSelector",
  component: DatasetSelector,
  argTypes: {
    allowCreation: {
      control: "boolean",
      description: "Allow creating new datasets",
    },
    placeholder: {
      control: "text",
      description: "Placeholder text when no dataset is selected",
    },
  },
  parameters: {
    controls: {
      exclude: ["onSelect", "selected", "className", "buttonProps"],
    },
  },
  args: {
    // Dummy onSelect to satisfy TypeScript - will be overridden in render
    onSelect: () => {},
  },
} satisfies Meta<typeof DatasetSelector>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select a dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelector
          {...args}
          selected={selected}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const EmptyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select a dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], []);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelector
          {...args}
          selected={selected}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const DisallowCreation: Story = {
  args: {
    allowCreation: false,
    placeholder: "Select a dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelector
          {...args}
          selected={selected}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

// TODO: we should handle extremely long lists (1000+) of datasets gracefully (e.g. truncate what we render)
export const ManyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select a dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    const repeatedMockDatasets = Array.from({ length: 100 }, (_, i) => ({
      name: `test_dataset_${i + 1}`,
      count: i + 1,
      lastUpdated: new Date(
        Date.now() - 1000 * 60 * 60 * (i + 1),
      ).toISOString(),
    }));
    queryClient.setQueryData(
      ["DATASETS_COUNT", undefined],
      repeatedMockDatasets,
    );

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelector
          {...args}
          selected={selected}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};
