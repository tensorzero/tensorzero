import InputSnippet from "./InputSnippet";
import type { Meta, StoryObj } from "@storybook/react-vite";
import pdfUrl from "./InputSnippet.stories.fixture.tensorzero.pdf?url";
import mp3Url from "./InputSnippet.stories.fixture.tensorzero.mp3?url";
import type { JsonValue } from "tensorzero-node";

const meta = {
  title: "InputSnippet",
  component: InputSnippet,
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
    messages: [],
  },
};

export const SystemNoMessages: Story = {
  args: {
    system: "You are a helpful assistant.",
    messages: [],
  },
};

export const MessagesNoSystem: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "What is the capital of Japan?\n\nRespond with just the city name.",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Tokyo",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Arigatou!",
          },
        ],
      },
    ],
  },
};

export const SingleUserMessageWithMultipleContentBlocks: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.",
          },
          {
            type: "text",
            text: "Duis sodales facilisis mollis. Sed et molestie enim. Integer eget pharetra urna. In ullamcorper nisi vitae ullamcorper laoreet. Vestibulum at enim et mauris tristique pellentesque. Sed dignissim nunc porta arcu sodales viverra. Nunc vulputate neque quis arcu ultricies, eu convallis magna tincidunt. Integer bibendum nec mauris ut mattis. Suspendisse potenti. Quisque gravida dui turpis. Duis vestibulum odio in risus finibus placerat.",
          },
          {
            type: "text",
            text: "Aliquam dapibus accumsan erat, eget volutpat mauris ultricies eu. Sed in tortor rutrum, scelerisque ipsum sit amet, volutpat ex. Ut sodales mauris ante, vitae condimentum elit euismod ac. Aliquam sed libero bibendum, venenatis lectus sed, pharetra diam. Ut eu viverra lacus. Fusce ornare vitae lectus ut ullamcorper. Mauris nec nisl convallis, tincidunt leo at, dignissim mi. Nam vehicula eleifend lectus eu scelerisque. Pellentesque feugiat eget risus sed posuere. Aliquam semper, enim eget consequat volutpat, felis sapien sagittis elit, condimentum gravida nisl ante sed eros. Vestibulum elementum efficitur mi, ac auctor lectus hendrerit vel. Quisque at enim libero. Cras in lectus vitae eros vestibulum mollis in et purus. Pellentesque tincidunt dui nec orci tincidunt, non fermentum felis molestie. Phasellus blandit, arcu quis interdum ultricies, turpis ligula tempor tellus, quis euismod est felis et tortor.",
          },
        ],
      },
    ],
  },
};

export const MultipleUserMessagesWithSingleContentBlock: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Duis sodales facilisis mollis. Sed et molestie enim. Integer eget pharetra urna. In ullamcorper nisi vitae ullamcorper laoreet. Vestibulum at enim et mauris tristique pellentesque. Sed dignissim nunc porta arcu sodales viverra. Nunc vulputate neque quis arcu ultricies, eu convallis magna tincidunt. Integer bibendum nec mauris ut mattis. Suspendisse potenti. Quisque gravida dui turpis. Duis vestibulum odio in risus finibus placerat.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Aliquam dapibus accumsan erat, eget volutpat mauris ultricies eu. Sed in tortor rutrum, scelerisque ipsum sit amet, volutpat ex. Ut sodales mauris ante, vitae condimentum elit euismod ac. Aliquam sed libero bibendum, venenatis lectus sed, pharetra diam. Ut eu viverra lacus. Fusce ornare vitae lectus ut ullamcorper. Mauris nec nisl convallis, tincidunt leo at, dignissim mi. Nam vehicula eleifend lectus eu scelerisque. Pellentesque feugiat eget risus sed posuere. Aliquam semper, enim eget consequat volutpat, felis sapien sagittis elit, condimentum gravida nisl ante sed eros. Vestibulum elementum efficitur mi, ac auctor lectus hendrerit vel. Quisque at enim libero. Cras in lectus vitae eros vestibulum mollis in et purus. Pellentesque tincidunt dui nec orci tincidunt, non fermentum felis molestie. Phasellus blandit, arcu quis interdum ultricies, turpis ligula tempor tellus, quis euismod est felis et tortor.",
          },
        ],
      },
    ],
  },
};

export const MultiTurnToolUse: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "What is the weather in Tokyo?",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "I can help you with that.",
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
            text: "The weather in Tokyo is sunny, with a temperature of 20 degrees Celsius.",
          },
        ],
      },
    ],
  },
};

export const LongMultiTurnToolUse: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "What is the weather in Tokyo?",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "I can help you with that.",
          },
          {
            type: "tool_call",
            name: "summarize_tool",
            arguments: JSON.stringify({
              verbosity: "high",
              text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et nunc augue. Pellentesque at facilisis ipsum. Donec facilisis lorem ligula, ultrices feugiat nibh consectetur id. Aenean pulvinar est ac ipsum vulputate, nec maximus ligula elementum. Cras a eros eget velit varius finibus ut sollicitudin enim. Nulla et augue ac massa consequat cursus. Curabitur eget dolor tristique, porttitor mi non, commodo augue. Integer tincidunt dui lectus, egestas dapibus mauris porta sit amet. Morbi tincidunt turpis id tortor ornare, vel viverra elit cursus. Cras a felis ultricies, interdum dui vel, facilisis risus.",
            }),
            id: "acd0806d-4ec6-4efd-864e-a29aa66ec3fc",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            name: "summarize_tool",
            result:
              "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et nunc augue. Pellentesque at facilisis ipsum. Donec facilisis lorem ligula, ultrices feugiat nibh consectetur id. Aenean pulvinar est ac ipsum vulputate, nec maximus ligula elementum. Cras a eros eget velit varius finibus ut sollicitudin enim. Nulla et augue ac massa consequat cursus. Curabitur eget dolor tristique, porttitor mi non, commodo augue. Integer tincidunt dui lectus, egestas dapibus mauris porta sit amet. Morbi tincidunt turpis id tortor ornare, vel viverra elit cursus. Cras a felis ultricies, interdum dui vel, facilisis risus.",
            id: "acd0806d-4ec6-4efd-864e-a29aa66ec3fc",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et nunc augue. Pellentesque at facilisis ipsum. Donec facilisis lorem ligula, ultrices feugiat nibh consectetur id. Aenean pulvinar est ac ipsum vulputate, nec maximus ligula elementum. Cras a eros eget velit varius finibus ut sollicitudin enim. Nulla et augue ac massa consequat cursus. Curabitur eget dolor tristique, porttitor mi non, commodo augue. Integer tincidunt dui lectus, egestas dapibus mauris porta sit amet. Morbi tincidunt turpis id tortor ornare, vel viverra elit cursus. Cras a felis ultricies, interdum dui vel, facilisis risus.",
          },
        ],
      },
    ],
  },
};

export const MultiTurnParallelToolUse: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "What is the weather in Tokyo?",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "I can help you with that.",
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
            text: "The weather in Tokyo is sunny, with a temperature of 20 degrees Celsius and a humidity of 50%.",
          },
        ],
      },
    ],
  },
};

export const RawTextInput: Story = {
  args: {
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
};

async function getBase64File(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch ${url}: ${response.status} ${response.statusText}`,
      );
    }
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        resolve(`data:${blob.type};base64,${base64String.split(",")[1]}`);
      };
      reader.onerror = () =>
        reject(new Error(`Failed to read file from ${url}`));
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    throw new Error(
      `Failed to load file from ${url}: ${error instanceof Error ? error.message : "Unknown error"}`,
    );
  }
}

export const ImageInput: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Do the images share any common features?",
          },
          {
            type: "file",
            file: {
              dataUrl: await getBase64File(
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
              path: "observability/files/e46e28c76498f7a7e935a502d3cd6f41052a76a6c6b0d8cda44e03fad8cc70f1.png",
            },
          },
          {
            type: "file",
            file: {
              // This is a one pixel by one pixel orange image
              dataUrl:
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",
              mime_type: "image/png",
            },
            storage_path: {
              kind: {
                type: "filesystem",
                path: "my_object_storage",
              },
              path: "observability/files/7f3a9b2e8d4c6f5a1e0b9d8c7a4b2e5f3d1c8b9a6e4f2d5c8b3a7e1f9d4c6b2.png",
            },
          },
        ],
      },
    ],
  },
};

export const ImageInputError: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Do the images share any common features?",
          },
          {
            type: "file_error",
            file: {
              mime_type: "image/png",
              url: "foo.png",
            },
            storage_path: {
              kind: {
                type: "s3_compatible",
                bucket_name: "tensorzero-e2e-test-images",
                region: "us-east-1",
                endpoint: null,
                allow_http: null,
              },
              path: "observability/files/e46e28c76498f7a7e935a502d3cd6f41052a76a6c6b0d8cda44e03fad8cc70f1.png",
            },
            error: "Failed to get object: Internal Server Error",
          },
          {
            type: "file_error",
            file: {
              mime_type: "image/png",
              url: "foo.png",
            },
            storage_path: {
              kind: {
                type: "s3_compatible",
                bucket_name: "tensorzero-e2e-test-images",
                region: "us-east-1",
                endpoint: null,
                allow_http: null,
              },
              path: "observability/files/e46e28c76498f7a7e935a502d3cd6f41052a76a6c6b0d8cda44e03fad8cc70f1.png",
            },
            error: "Failed to get object: Timeout",
          },
        ],
      },
    ],
  },
};

export const PDFInput: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Please analyze this research paper.",
          },
          {
            type: "file",
            file: {
              dataUrl: await getBase64File(pdfUrl),
              mime_type: "application/pdf",
            },
            storage_path: {
              kind: {
                type: "s3_compatible",
                bucket_name: "tensorzero-documents",
                region: "us-east-1",
                endpoint: null,
                allow_http: null,
              },
              path: "observability/files/contract_fake_path_12345.pdf",
            },
          },
        ],
      },
    ],
  },
};

export const AudioInput: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Transcribe this audio recording.",
          },
          {
            type: "file",
            file: {
              dataUrl: await getBase64File(mp3Url),
              mime_type: "audio/mp3",
            },
            storage_path: {
              kind: {
                type: "filesystem",
                path: "my_audio_storage",
              },
              path: "observability/files/meeting_recording_fake_path_67890.mp3",
            },
          },
        ],
      },
    ],
  },
};

export const BadToolInput: Story = {
  args: {
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool_call",
            name: "search_wikipedia",
            arguments: '{"query":"Adolescent cliques social groups peer ',
            id: "toolu_011BJUPo2bXLyXMjzfDhZFVx",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            name: "bad_name",
            result:
              "Peer group\nClique\nAdolescent clique\nPeer pressure\nAdolescence\nElaboration principle\nSocial emotional development\nSocial software\nCrowds (adolescence)\nSocial network analysis",
            id: "bad_id",
          },
        ],
      },
    ],
  },
};

export const TextIsJSON: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: '"All these content blocks are string literals!"',
          },
          // This one is a string literal
          {
            type: "text",
            text: '{"key": "value"}',
          },
          {
            type: "text",
            text: "[1, 2, 3]",
          },
          {
            type: "text",
            text: "123",
          },

          {
            type: "text",
            text: "null",
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: '"All these content blocks are string literals!"',
          },
          {
            type: "text",
            text: '{"key": "value"}',
          },
          {
            type: "text",
            text: "[1, 2, 3]",
          },
          {
            type: "text",
            text: "123",
          },

          {
            type: "text",
            text: "null",
          },
        ],
      },
    ],
  },
};

export const UnknownAndThoughtContent: Story = {
  args: {
    messages: [
      {
        role: "user",
        content: [
          {
            type: "unknown",
            data: null,
            model_provider_name: null,
          },
          {
            type: "unknown",
            data: {
              some: "arbitrary",
              data: 123,
              structure: ["is", "not", "validated"],
            } as JsonValue,
            model_provider_name: null,
          },
          {
            type: "thought",
            text: "This is a thought content block for testing.",
            signature: undefined,
            _internal_provider_type: undefined,
          },
        ],
      },
    ],
  },
};

export const TemplateInput: Story = {
  args: {
    system:
      "You are a helpful assistant that responds to prompts generated from templates.",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "template",
            name: "question_answering",
            arguments: {
              question: "What is the capital of France?",
              context:
                "France is a country in Western Europe with Paris as its capital and largest city.",
              format: "brief",
            },
          },
        ],
      },
    ],
  },
};
