import type { Meta, StoryObj } from "@storybook/react-vite";
import { ToolDetails } from "./ToolDetails";
import type { Config } from "tensorzero-node";
import { useState } from "react";
import { ConfigProvider } from "~/context/config";
import { Button } from "../ui/button";

const meta = {
  title: "UI/ToolDetails",
  component: ToolDetails,
  decorators: [
    (Story) => (
      <div className="w-sm p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ToolDetails>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockConfig = {
  tools: {
    search_wikipedia: {
      name: "search_wikipedia",
      description:
        "Search Wikipedia for pages that match the query. Returns a list of page titles.",
      parameters: {
        value: {
          $schema: "http://json-schema.org/draft-07/schema#",
          type: "object",
          description:
            "Search Wikipedia for pages that match the query. Returns a list of page titles.",
          properties: {
            query: {
              type: "string",
              description:
                'The query to search Wikipedia for (e.g. "machine learning").',
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
      },
      strict: true,
    },
  },
} as unknown as Config;

export const Default: Story = {
  args: {
    toolName: null,
  },
  render: function DefaultStory(args) {
    const [tool, setTool] = useState<string | null>(args.toolName);

    return (
      <ConfigProvider value={mockConfig}>
        <Button onClick={() => setTool("search_wikipedia")}>Open</Button>
        <ToolDetails toolName={tool} onClose={() => setTool(null)} />
      </ConfigProvider>
    );
  },
};
