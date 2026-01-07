import type { Meta, StoryObj } from "@storybook/react-vite";
import { RouteErrorContent } from "./RouteErrorContent";

const meta = {
  title: "Error/RouteErrorContent",
  component: RouteErrorContent,
  parameters: {
    layout: "fullscreen",
  },
  decorators: [
    (Story) => (
      <div className="bg-background min-h-screen">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof RouteErrorContent>;

export default meta;
type Story = StoryObj<typeof meta>;

export const GenericError: Story = {
  args: {
    error: new Error("Failed to load data from the server"),
  },
};

export const NotFound: Story = {
  args: {
    error: {
      status: 404,
      statusText: "Not Found",
      data: "The requested resource could not be found.",
    },
  },
};

export const ServerError: Story = {
  args: {
    error: {
      status: 500,
      statusText: "Internal Server Error",
      data: "An unexpected error occurred while processing your request.",
    },
  },
};

export const BadRequest: Story = {
  args: {
    error: {
      status: 400,
      statusText: "Bad Request",
      data: "The request was malformed or contained invalid parameters.",
    },
  },
};

export const UnknownError: Story = {
  args: {
    error: "Something went wrong",
  },
};
