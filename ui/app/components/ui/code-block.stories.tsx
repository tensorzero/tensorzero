import type { Meta, StoryObj } from "@storybook/react-vite";
import { codeToHtml } from "shiki";
import { CodeBlock } from "./code-block";

const meta = {
  title: "CodeBlock",
  component: CodeBlock,
} satisfies Meta<typeof CodeBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithRawCode: Story = {
  args: {
    showLineNumbers: true,
    raw: `function helloWorld() {
	console.log("Hello, world!");

    return function goodbye() {
		console.log("Goodbye!");
	}
}`,
  },
};

export const WithHtml: Story = {
  args: {
    showLineNumbers: true,
    html: `<pre tabindex="0"><code><span class="line">function helloWorld() {</span>\n<span class="line">	console.log("Hello, world!");</span>\n<span class="line">}</span></code></pre>`,
  },
};

export const WithSyntaxHighlighting: Story = {
  name: "With Shiki syntax highlighting",
  args: {
    showLineNumbers: true,
    html: await codeToHtml(
      JSON.stringify(
        {
          name: "tensorzero-ui",
          private: true,
          type: "module",
          scripts: {
            build: "NODE_ENV=production react-router build",
            dev: "react-router dev",
          },
          dependencies: {
            "@clickhouse/client": "^1.11.0",
          },
          devDependencies: {
            tailwindcss: "^4.1.2",
            typescript: "^5.8.2",
            vite: "^6.2.7",
          },
        },
        null,
        2,
      ),
      {
        lang: "json",
        theme: "github-light",
      },
    ),
  },
};
