import InputSnippet from "./InputSnippet";
import type { Meta, StoryObj } from "@storybook/react";
import { withRouter } from "storybook-addon-remix-react-router";

const meta = {
  title: "InputSnippet",
  component: InputSnippet,
  decorators: [withRouter],
  render: (args) => (
    <div className="w-[80vw] p-4">
      <InputSnippet {...args} />
    </div>
  ),
} satisfies Meta<typeof InputSnippet>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    input: {
      messages: [],
    },
  },
};

export const SystemNoMessages: Story = {
  args: {
    input: {
      system: "You are a helpful assistant.",
      messages: [],
    },
  },
};

export const MessagesNoSystem: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "What is the capital of Japan?",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value: "The capital of Japan is Tokyo.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "Arigatou!",
            },
          ],
        },
      ],
    },
  },
};

export const SingleUserMessageWithMultipleContentBlocks: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value:
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.",
            },
            {
              type: "text",
              value:
                "Duis sodales facilisis mollis. Sed et molestie enim. Integer eget pharetra urna. In ullamcorper nisi vitae ullamcorper laoreet. Vestibulum at enim et mauris tristique pellentesque. Sed dignissim nunc porta arcu sodales viverra. Nunc vulputate neque quis arcu ultricies, eu convallis magna tincidunt. Integer bibendum nec mauris ut mattis. Suspendisse potenti. Quisque gravida dui turpis. Duis vestibulum odio in risus finibus placerat.",
            },
            {
              type: "text",
              value:
                "Aliquam dapibus accumsan erat, eget volutpat mauris ultricies eu. Sed in tortor rutrum, scelerisque ipsum sit amet, volutpat ex. Ut sodales mauris ante, vitae condimentum elit euismod ac. Aliquam sed libero bibendum, venenatis lectus sed, pharetra diam. Ut eu viverra lacus. Fusce ornare vitae lectus ut ullamcorper. Mauris nec nisl convallis, tincidunt leo at, dignissim mi. Nam vehicula eleifend lectus eu scelerisque. Pellentesque feugiat eget risus sed posuere. Aliquam semper, enim eget consequat volutpat, felis sapien sagittis elit, condimentum gravida nisl ante sed eros. Vestibulum elementum efficitur mi, ac auctor lectus hendrerit vel. Quisque at enim libero. Cras in lectus vitae eros vestibulum mollis in et purus. Pellentesque tincidunt dui nec orci tincidunt, non fermentum felis molestie. Phasellus blandit, arcu quis interdum ultricies, turpis ligula tempor tellus, quis euismod est felis et tortor.",
            },
          ],
        },
      ],
    },
  },
};

export const MultipleUserMessagesWithSingleContentBlock: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value:
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              value:
                "Duis sodales facilisis mollis. Sed et molestie enim. Integer eget pharetra urna. In ullamcorper nisi vitae ullamcorper laoreet. Vestibulum at enim et mauris tristique pellentesque. Sed dignissim nunc porta arcu sodales viverra. Nunc vulputate neque quis arcu ultricies, eu convallis magna tincidunt. Integer bibendum nec mauris ut mattis. Suspendisse potenti. Quisque gravida dui turpis. Duis vestibulum odio in risus finibus placerat.",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              value:
                "Aliquam dapibus accumsan erat, eget volutpat mauris ultricies eu. Sed in tortor rutrum, scelerisque ipsum sit amet, volutpat ex. Ut sodales mauris ante, vitae condimentum elit euismod ac. Aliquam sed libero bibendum, venenatis lectus sed, pharetra diam. Ut eu viverra lacus. Fusce ornare vitae lectus ut ullamcorper. Mauris nec nisl convallis, tincidunt leo at, dignissim mi. Nam vehicula eleifend lectus eu scelerisque. Pellentesque feugiat eget risus sed posuere. Aliquam semper, enim eget consequat volutpat, felis sapien sagittis elit, condimentum gravida nisl ante sed eros. Vestibulum elementum efficitur mi, ac auctor lectus hendrerit vel. Quisque at enim libero. Cras in lectus vitae eros vestibulum mollis in et purus. Pellentesque tincidunt dui nec orci tincidunt, non fermentum felis molestie. Phasellus blandit, arcu quis interdum ultricies, turpis ligula tempor tellus, quis euismod est felis et tortor.",
            },
          ],
        },
      ],
    },
  },
};

export const MultiTurnToolUse: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [{ type: "text", value: "What is the weather in Tokyo?" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value: "I can help you with that.",
            },
            {
              type: "tool_call",
              name: "weather_tool",
              arguments: JSON.stringify({ city: "Tokyo" }),
              id: "acd0806d-4ec6-4efd-864e-a29aa66ec3fc",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "weather_tool",
              result: JSON.stringify({ temperature: 20, condition: "sunny" }),
              id: "acd0806d-4ec6-4efd-864e-a29aa66ec3fc",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value:
                "The weather in Tokyo is sunny, with a temperature of 20 degrees Celsius.",
            },
          ],
        },
      ],
    },
  },
};

export const MultiTurnParallelToolUse: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [{ type: "text", value: "What is the weather in Tokyo?" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value: "I can help you with that.",
            },
            {
              type: "tool_call",
              name: "temperature_tool",
              arguments: JSON.stringify({ city: "Tokyo" }),
              id: "1",
            },
            {
              type: "tool_call",
              name: "humidity_tool",
              arguments: JSON.stringify({ city: "Tokyo" }),
              id: "2 ",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "temperature_tool",
              result: JSON.stringify({ temperature: 20 }),
              id: "1",
            },
            {
              type: "tool_result",
              name: "humidity_tool",
              result: JSON.stringify({ humidity: 50 }),
              id: "2",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value:
                "The weather in Tokyo is sunny, with a temperature of 20 degrees Celsius and a humidity of 50%.",
            },
          ],
        },
      ],
    },
  },
};

export const StructuredInputs: Story = {
  args: {
    input: {
      system: "Write a haiku about the topic provided by the user.",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: JSON.stringify({ topic: "AI" }),
            },
          ],
        },
      ],
    },
  },
};

export const RawTextInput: Story = {
  args: {
    input: {
      system: "Write a haiku about the topic provided by the user.",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "raw_text",
              value: "# Topic\n\nAI",
            },
          ],
        },
      ],
    },
  },
};

async function getBase64Image(url: string): Promise<string> {
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      resolve(`data:${blob.type};base64,${base64String.split(",")[1]}`);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export const ImageInput: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "Do the images share any common features?",
            },
            {
              type: "image",
              image: {
                url: await getBase64Image(
                  "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                ),
                mime_type: "image/png",
              },
              storage_path: {
                kind: {
                  type: "s3_compatible",
                  bucket_name: "tensorzero-e2e-test-images",
                  region: "us-east-1",
                  endpoint: null,
                  allow_http: null,
                },
                path: "observability/images/e46e28c76498f7a7e935a502d3cd6f41052a76a6c6b0d8cda44e03fad8cc70f1.png",
              },
            },
            {
              type: "image",
              image: {
                // This is a one pixel by one pixel orange image
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",
                mime_type: "image/png",
              },
              storage_path: {
                kind: {
                  type: "filesystem",
                  path: "my_object_storage",
                },
                path: "observability/images/7f3a9b2e8d4c6f5a1e0b9d8c7a4b2e5f3d1c8b9a6e4f2d5c8b3a7e1f9d4c6b2.png",
              },
            },
          ],
        },
      ],
    },
  },
};

export const ImageInputError: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "Do the images share any common features?",
            },
            {
              type: "image_error",
              error: "Failed to get object: Internal Server Error",
            },
            {
              type: "image_error",
              error: "Failed to get object: Timeout",
            },
          ],
        },
      ],
    },
  },
};
