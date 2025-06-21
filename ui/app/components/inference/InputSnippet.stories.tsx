import InputSnippet from "./InputSnippet";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import pdfUrl from "./InputSnippet.stories.fixture.tensorzero.pdf?url";
import mp3Url from "./InputSnippet.stories.fixture.tensorzero.mp3?url";

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

export const LongMultiTurnToolUse: Story = {
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
              result: JSON.stringify({
                summary:
                  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et nunc augue. Pellentesque at facilisis ipsum. Donec facilisis lorem ligula, ultrices feugiat nibh consectetur id. Aenean pulvinar est ac ipsum vulputate, nec maximus ligula elementum. Cras a eros eget velit varius finibus ut sollicitudin enim. Nulla et augue ac massa consequat cursus. Curabitur eget dolor tristique, porttitor mi non, commodo augue. Integer tincidunt dui lectus, egestas dapibus mauris porta sit amet. Morbi tincidunt turpis id tortor ornare, vel viverra elit cursus. Cras a felis ultricies, interdum dui vel, facilisis risus.",
              }),
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
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam et nunc augue. Pellentesque at facilisis ipsum. Donec facilisis lorem ligula, ultrices feugiat nibh consectetur id. Aenean pulvinar est ac ipsum vulputate, nec maximus ligula elementum. Cras a eros eget velit varius finibus ut sollicitudin enim. Nulla et augue ac massa consequat cursus. Curabitur eget dolor tristique, porttitor mi non, commodo augue. Integer tincidunt dui lectus, egestas dapibus mauris porta sit amet. Morbi tincidunt turpis id tortor ornare, vel viverra elit cursus. Cras a felis ultricies, interdum dui vel, facilisis risus.",
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

export const TextWithArguments: Story = {
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
              type: "file",
              file: {
                url: await getBase64File(
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
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",
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
              type: "file_error",
              error: "Failed to get object: Internal Server Error",
            },
            {
              type: "file_error",
              error: "Failed to get object: Timeout",
            },
          ],
        },
      ],
    },
  },
};

export const PDFInput: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "Please analyze this research paper.",
            },
            {
              type: "file",
              file: {
                url: await getBase64File(pdfUrl),
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
  },
};

export const AudioInput: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value: "Transcribe this audio recording.",
            },
            {
              type: "file",
              file: {
                url: await getBase64File(mp3Url),
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
  },
};

export const BadToolInput: Story = {
  args: {
    input: {
      system: {
        question:
          "Do adolescents associate with 'cliques' on a large or small scale?",
      },
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              value:
                'Do adolescents associate with "cliques" on a large or small scale?',
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              value:
                'Compact Research Summary: Do Adolescents Associate with "Cliques" on a Large or Small Scale?\n\nKey Findings:\n1. Scale of Clique Association:\n- Cliques typically consist of 5-6 people\n- Most adolescents fall into three categories:\n  a) Group members (less than half of school population)\n  b) Liaisons (connect across multiple groups)\n  c) Isolates (few or no close peer relationships)\n\n2. Prevalence:\n- Cliques are extremely common during adolescence\n- Form in early adolescence (girls around 11, boys around 13-14)\n- Present across different demographics, ethnicities, and socioeconomic backgrounds\n\n3. Characteristics:\n- Most cliques are homogeneous in:\n  - Age\n  - Gender\n  - Race\n  - Social status\n  - Socioeconomic background\n\n4. Duration and Stability:\n- Cliques are relatively stable, losing about one-third of members per year\n- Membership type (group member, liaison, isolate) is more consistent than specific clique membership\n- Typically dissolve by late adolescence, replaced by dating relationships\n\n5. Demographic Factors:\n- Strongly influenced by:\n  - School environment\n  - Shared interests\n  - Orientation toward school/teen culture\n  - Potential antisocial behaviors\n\nConclusion: Adolescents do associate with cliques on a moderate to large scale, with most students belonging to some form of social group during their teenage years.',
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              name: "think",
              arguments:
                '{"thought":"I notice that my previous response was generated without using Wikipedia search tools, which goes against the established protocol. I should systematically search Wikipedia to provide a well-sourced answer about adolescent cliques."}',
              id: "toolu_01B8b7WQWmT2XHE6AQo1Ms7E",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "think",
              result: "",
              id: "toolu_01B8b7WQWmT2XHE6AQo1Ms7E",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              name: "search_wikipedia",
              arguments:
                '{"query":"Adolescent cliques social groups peer relationships"}',
              id: "toolu_011BJUPo2bXLyXMjzfDhZFVx",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "search_wikipedia",
              result:
                "Peer group\nClique\nAdolescent clique\nPeer pressure\nAdolescence\nElaboration principle\nSocial emotional development\nSocial software\nCrowds (adolescence)\nSocial network analysis",
              id: "toolu_011BJUPo2bXLyXMjzfDhZFVx",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              name: "think",
              arguments:
                '{"thought":"The search results provide several promising pages to investigate. I\'ll start by loading the \\"Adolescent clique\\" page to get direct information about the scale and nature of cliques among adolescents."}',
              id: "toolu_01UbpVHwqoY3QvUBmVE32HdA",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "think",
              result: "",
              id: "toolu_01UbpVHwqoY3QvUBmVE32HdA",
            },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              name: "load_wikipedia_page",
              arguments: '{"title":"Adolescent clique"}',
              id: "toolu_019Wm8MrB7ECsV3ubbce9MJf",
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              name: "load_wikipedia_page",
              result:
                '# URL\n\nhttps://en.wikipedia.org/wiki/Adolescent_clique\n\n# CONTENT\n\n|  |  |\n| --- | --- |\n|  | This article needs to be **updated**. The reason given is: article is largely based on late 2000s and early 2010s conceptions of cliques (for example, the inclusion of virtually non-existent cliques that used to be prominent like skaters). Please help update this article to reflect recent events or newly available information. *(March 2023)* |\n\nCliques that develop among adolescents\n\n**Adolescent cliques** are [cliques](/wiki/Clique "Clique") that develop amongst [adolescents](/wiki/Adolescence "Adolescence"). In the social sciences, the word "**clique**" is used to describe a large group of 6 to 12 "who interact with each other more regularly and intensely than others in the same setting".[[1]](#cite_note-EEP-1) Cliques are distinguished from "[crowds](/wiki/Crowds_(adolescence) "Crowds (adolescence)")" in that their members socially interact with one another more than the typical crowd (e.g. hang out together, go shopping, play sports etc.). Crowds, on the other hand, are defined by reputation. Although the word \'clique\' or \'cliquey\' is often used in day-to-day conversation to describe [relational aggression](/wiki/Relational_aggression "Relational aggression") or snarky, gossipy behaviors of groups of socially dominant teenage girls, that is not always accurate.[[2]](#cite_note-2) Interacting with cliques is part of normative social development regardless of gender, ethnicity, or popularity. Although cliques are most commonly studied during [adolescence](/wiki/Adolescence "Adolescence") and in educational settings, they can exist in all age groups and settings.\n\nContents\n--------\n\n* [1 Definition](#Definition)\n* [2 Clique membership](#Clique_membership)\n  + [2.1 Common misconceptions](#Common_misconceptions)\n  + [2.2 Forms of association](#Forms_of_association)\n  + [2.3 Stability over time](#Stability_over_time)\n* [3 Within clique structure](#Within_clique_structure)\n  + [3.1 Popularity](#Popularity)\n* [4 Between clique structure](#Between_clique_structure)\n  + [4.1 Personal factors](#Personal_factors)\n    - [4.1.1 Orientation toward school](#Orientation_toward_school)\n    - [4.1.2 Orientation toward teen culture](#Orientation_toward_teen_culture)\n    - ',
              id: "toolu_019Wm8MrB7ECsV3ubbce9MJf",
            },
          ],
        },
      ],
    },
  },
};
