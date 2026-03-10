import type { Meta, StoryObj } from "@storybook/react-vite";
import { AutoevalExampleLabelingCard } from "~/components/autopilot/AutoevalExampleLabeling";
import type { EventPayloadAutoevalExampleLabeling } from "~/types/tensorzero";

// ── Fixtures ──────────────────────────────────────────────────────────

const labelingOptions = [
  {
    id: "yes",
    label: "Yes",
    description: "The response exhibits the target behavior.",
  },
  {
    id: "no",
    label: "No",
    description: "The response does not exhibit the target behavior.",
  },
  {
    id: "irrelevant",
    label: "Irrelevant",
    description: "This example is not relevant to the target behavior.",
  },
];

const labelingPayload: EventPayloadAutoevalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "What's the capital of France?",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "The capital of France is Paris.",
        },
      ],
      questions: [
        {
          id: "ex1-label",
          header: "Example 1",
          question: "Does this example exhibit the target behavior?",
          type: "multiple_choice",
          options: labelingOptions,
          multi_select: false,
        },
        {
          id: "ex1-explanation",
          header: "Explanation 1",
          question: "Briefly explain your label (optional).",
          type: "free_response",
        },
      ],
    },
    {
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "I'm working on a research paper about the solar system and I need detailed, accurate information. Can you tell me about the largest planet in our solar system? I'd like to know about its size relative to other planets, its composition, its moons, its ring system, and any notable features like the Great Red Spot. Also, please explain how it was discovered and named, and what role it plays in protecting the inner solar system from asteroid impacts. Finally, describe any recent space missions that have studied this planet.",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "The largest planet in our solar system is Saturn. It is a gas giant composed primarily of hydrogen and helium. Saturn is the sixth planet from the Sun and is best known for its prominent ring system, which is made up of ice particles, rocky debris, and dust. The planet has a diameter of about 116,460 kilometers, making it the second-largest planet in our solar system after Jupiter.\n\nSaturn has at least 146 known moons, with Titan being the largest. Titan is notable for having a thick atmosphere and liquid hydrocarbon lakes on its surface. Another notable moon is Enceladus, which has geysers of water ice erupting from its south pole, suggesting a subsurface ocean.\n\nThe planet was known to ancient civilizations and was named after the Roman god of agriculture and time. Galileo first observed Saturn through a telescope in 1610, though he didn't recognize its rings as such.\n\nRecent missions include NASA's Cassini-Huygens mission, which orbited Saturn from 2004 to 2017 and provided unprecedented data about the planet, its rings, and its moons. The Huygens probe landed on Titan in 2005, marking the first landing in the outer solar system.",
        },
      ],
      questions: [
        {
          id: "ex2-label",
          header: "Example 2",
          question: "Does this example exhibit the target behavior?",
          type: "multiple_choice",
          options: labelingOptions,
          multi_select: false,
        },
        {
          id: "ex2-explanation",
          header: "Explanation 2",
          question: "Briefly explain your label (optional).",
          type: "free_response",
        },
      ],
    },
    {
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "Who wrote Romeo and Juliet?",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "Romeo and Juliet was written by William Shakespeare, believed to have been composed between 1594 and 1596. It is one of the most famous love stories in the English language and has been adapted countless times for stage, film, and other media.\n\nThe play tells the story of two young lovers from feuding families in Verona, Italy — the Montagues and the Capulets. Their secret romance ultimately ends in tragedy when a series of misunderstandings leads both to take their own lives.\n\nShakespeare drew inspiration from earlier sources, most notably Arthur Brooke's narrative poem 'The Tragical History of Romeus and Juliet' (1562), which itself was based on Italian sources. However, Shakespeare transformed the material significantly, adding memorable characters like Mercutio and the Nurse, and elevating the language to some of the most beautiful poetry ever written in English.\n\nThe play was first published in quarto form in 1597, though this version is considered a 'bad quarto' — likely reconstructed from memory by actors. A more authoritative second quarto appeared in 1599.\n\nRomeo and Juliet remains one of Shakespeare's most frequently performed plays and has influenced Western culture's conception of romantic love profoundly.",
        },
      ],
      questions: [
        {
          id: "ex3-label",
          header: "Example 3",
          question: "Does this example exhibit the target behavior?",
          type: "multiple_choice",
          options: labelingOptions,
          multi_select: false,
        },
        {
          id: "ex3-explanation",
          header: "Explanation 3",
          question: "Briefly explain your label (optional).",
          type: "free_response",
        },
      ],
    },
  ],
};

// ── Story ────────────────────────────────────────────────────────────

const meta = {
  title: "Autopilot/ExampleLabeling",
  render: () => (
    <div className="w-[600px] p-4">
      <AutoevalExampleLabelingCard payload={labelingPayload} />
    </div>
  ),
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
