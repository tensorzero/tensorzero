import type { Meta, StoryObj } from "@storybook/react-vite";
import DatasetSelector from "./DatasetSelector";
import { useArgs } from "storybook/preview-api";
import {
  createHookMockDecorator,
  createDatasetCountFetcherMock,
} from "../../../.storybook/mock-utils";

// Create a mock module object to avoid importing server-side code
const DatasetCountsModule = {
  useDatasetCountFetcher: () => ({ datasets: null, isLoading: false }),
};

// Ordered incorrectly to test sorting
const mockDatasets = [
  {
    dataset_name: "test_dataset",
    count: 3250,
    last_updated: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
  },
  {
    dataset_name: "evaluation_set",
    count: 850,
    last_updated: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
  },
  {
    dataset_name: "training_data_v2",
    count: 42100,
    last_updated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(), // 3 days ago
  },
  {
    dataset_name: "validation_set",
    count: 1200,
    last_updated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week ago
  },
  {
    dataset_name: "production_data",
    count: 15420,
    last_updated: new Date(Date.now() - 1000 * 60 * 5).toISOString(), // 5 minutes ago
  },
  {
    dataset_name:
      "long_dataset_name_that_should_still_be_displayed_gracefully_somehow",
    count: 1,
    last_updated: new Date(Date.now()).toISOString(), // now
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
  decorators: [
    createHookMockDecorator({
      module: DatasetCountsModule,
      hookName: "useDatasetCountFetcher",
      mockImplementation: createDatasetCountFetcherMock(mockDatasets),
    }),
  ],
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <DatasetSelector
        {...args}
        selected={selected}
        onSelect={(dataset, isNew) => {
          updateArgs({ selected: dataset });
          console.log(`Selected: ${dataset}, isNew: ${isNew}`);
        }}
      />
    );
  },
};

export const EmptyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select a dataset",
  },
  decorators: [
    createHookMockDecorator({
      module: DatasetCountsModule,
      hookName: "useDatasetCountFetcher",
      mockImplementation: createDatasetCountFetcherMock([]),
    }),
  ],
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <DatasetSelector
        {...args}
        selected={selected}
        onSelect={(dataset, isNew) => {
          updateArgs({ selected: dataset });
          console.log(`Selected: ${dataset}, isNew: ${isNew}`);
        }}
      />
    );
  },
};

export const DisallowCreation: Story = {
  args: {
    allowCreation: false,
    placeholder: "Select a dataset",
  },
  decorators: [
    createHookMockDecorator({
      module: DatasetCountsModule,
      hookName: "useDatasetCountFetcher",
      mockImplementation: createDatasetCountFetcherMock(mockDatasets),
    }),
  ],
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <DatasetSelector
        {...args}
        selected={selected}
        onSelect={(dataset, isNew) => {
          updateArgs({ selected: dataset });
          console.log(`Selected: ${dataset}, isNew: ${isNew}`);
        }}
      />
    );
  },
};

// TODO: we should handle extremely long lists (1000+) of datasets gracefully (e.g. truncate what we render)
export const ManyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select a dataset",
  },
  decorators: [
    createHookMockDecorator({
      module: DatasetCountsModule,
      hookName: "useDatasetCountFetcher",
      mockImplementation: createDatasetCountFetcherMock(
        Array.from({ length: 100 }, (_, i) => ({
          dataset_name: `test_dataset_${i + 1}`,
          count: i + 1,
          last_updated: new Date(
            Date.now() - 1000 * 60 * 60 * (i + 1),
          ).toISOString(),
        })),
      ),
    }),
  ],
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();

    return (
      <DatasetSelector
        {...args}
        selected={selected}
        onSelect={(dataset, isNew) => {
          updateArgs({ selected: dataset });
          console.log(`Selected: ${dataset}, isNew: ${isNew}`);
        }}
      />
    );
  },
};
