import type { Meta, StoryObj } from "@storybook/react-vite";
import { DatasetCombobox } from "./DatasetCombobox";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useArgs } from "storybook/preview-api";
import { mockDatasets, createManyDatasets } from "./dataset-stories-fixtures";

const meta = {
  title: "Dataset/DatasetCombobox",
  component: DatasetCombobox,
  argTypes: {
    allowCreation: {
      control: "boolean",
      description: "Allow creating new datasets",
    },
    disabled: {
      control: "boolean",
      description: "Disable the combobox",
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
} satisfies Meta<typeof DatasetCombobox>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    allowCreation: true,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected={selected ?? null}
            onSelect={(dataset) => updateArgs({ selected: dataset })}
          />
        </div>
      </QueryClientProvider>
    );
  },
};

export const WithSelection: Story = {
  args: {
    allowCreation: true,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected={selected ?? "test_dataset"}
            onSelect={(dataset) => updateArgs({ selected: dataset })}
          />
        </div>
      </QueryClientProvider>
    );
  },
};

export const EmptyDatasets: Story = {
  args: {
    allowCreation: true,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], []);

    return (
      <QueryClientProvider client={queryClient}>
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected={selected ?? null}
            onSelect={(dataset) => updateArgs({ selected: dataset })}
          />
        </div>
      </QueryClientProvider>
    );
  },
};

export const DisallowCreation: Story = {
  args: {
    allowCreation: false,
  },
  render: function Render(args) {
    const [{ selected }, updateArgs] = useArgs<{ selected?: string }>();
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected={selected ?? null}
            onSelect={(dataset) => updateArgs({ selected: dataset })}
          />
        </div>
      </QueryClientProvider>
    );
  },
};

export const Disabled: Story = {
  args: {
    allowCreation: true,
    disabled: true,
  },
  render: function Render(args) {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["DATASETS_COUNT", undefined], mockDatasets);

    return (
      <QueryClientProvider client={queryClient}>
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected="test_dataset"
            onSelect={() => {}}
          />
        </div>
      </QueryClientProvider>
    );
  },
};

// TODO: we should handle extremely long lists (1000+) of datasets gracefully (e.g. virtualization)
export const ManyDatasets: Story = {
  args: {
    allowCreation: true,
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
        <div className="w-80">
          <DatasetCombobox
            {...args}
            selected={selected ?? null}
            onSelect={(dataset) => updateArgs({ selected: dataset })}
          />
        </div>
      </QueryClientProvider>
    );
  },
};
