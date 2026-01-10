import type { Meta, StoryObj } from "@storybook/react-vite";
import { PageErrorContent } from "./ErrorContent";

const meta = {
  title: "Error/PageErrorContent",
  component: PageErrorContent,
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
} satisfies Meta<typeof PageErrorContent>;

export default meta;
type Story = StoryObj<typeof meta>;

// Helper to create mock RouteErrorResponse objects that pass isRouteErrorResponse()
function createMockRouteErrorResponse(
  status: number,
  data: unknown,
  statusText = "",
) {
  return { status, statusText, data, internal: true };
}

export const GenericError: Story = {
  args: {
    error: new Error("Failed to load data from the server"),
  },
};

export const NotFound: Story = {
  args: {
    error: createMockRouteErrorResponse(
      404,
      "The requested resource could not be found.",
      "Not Found",
    ),
  },
};

export const ServerError: Story = {
  args: {
    error: createMockRouteErrorResponse(
      500,
      "An unexpected error occurred while processing your request.",
      "Internal Server Error",
    ),
  },
};

export const BadRequest: Story = {
  args: {
    error: createMockRouteErrorResponse(
      400,
      "The request was malformed or contained invalid parameters.",
      "Bad Request",
    ),
  },
};

export const UnknownError: Story = {
  args: {
    error: "Something went wrong",
  },
};
