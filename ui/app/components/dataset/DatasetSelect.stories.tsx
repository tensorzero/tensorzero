import type { Meta, StoryObj } from "@storybook/react-vite";
import { DatasetSelect } from "./DatasetSelect";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useArgs } from "storybook/preview-api";
import { mockDatasets, createManyDatasets } from "./dataset-stories-fixtures";

const meta = {
  title: "Dataset/DatasetSelect",
  component: DatasetSelect,
  argTypes: {
    allowCreation: {
      control: "boolean",
      description: "Allow creating new datasets",
    },
    placeholder: {
      control: "text",
      description: "Placeholder text when no dataset is selected",
    },
    disabled: {
      control: "boolean",
      description: "Disable the select",
    },
  },
  parameters: {
    controls: {
      exclude: ["onSelect", "selected", "functionName"],
    },
  },
  args: {
    onSelect: () => {},
    selected: null,
  },
} satisfies Meta<typeof DatasetSelect>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect
          {...args}
          selected={selected ?? null}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const WithSelection: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect
          {...args}
          selected={selected ?? "test_dataset"}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const EmptyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], []);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect
          {...args}
          selected={selected ?? null}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const DisallowCreation: Story = {
  args: {
    allowCreation: false,
    placeholder: "Select dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect
          {...args}
          selected={selected ?? null}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};

export const Disabled: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select dataset",
    disabled: true,
  },
  render: function Render(args) {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect {...args} selected="test_dataset" onSelect={() => {}} />
      </QueryClientProvider>
    );
  },
};

// TODO: we should handle extremely long lists (1000+) of datasets gracefully (e.g. virtualization)
export const ManyDatasets: Story = {
  args: {
    allowCreation: true,
    placeholder: "Select dataset",
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(
      ["DATASETS_COUNT", undefined],
      createManyDatasets(100),
    );

    return (
      <QueryClientProvider client={queryClient}>
        <DatasetSelect
          {...args}
          selected={selected ?? null}
          onSelect={(dataset) => updateArgs({ selected: dataset })}
        />
      </QueryClientProvider>
    );
  },
};
