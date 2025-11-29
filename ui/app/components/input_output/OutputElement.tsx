import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "~/types/tensorzero";
import { ChatOutputElement } from "./ChatOutputElement";
import { JsonOutputElement } from "./JsonOutputElement";

interface OutputElementProps {
  type: "chat" | "json";
  output?: ContentBlockChatOutput[] | JsonInferenceOutput;
  isEditing?: boolean;
  onOutputChange?: (
    output?: ContentBlockChatOutput[] | JsonInferenceOutput,
  ) => void;
  maxHeight?: number | "Content";
}

export function OutputElement({ type, output, ...props }: OutputElementProps) {
  switch (type) {
    case "chat":
      return (
        <ChatOutputElement
          output={output as ContentBlockChatOutput[] | undefined}
          {...props}
        />
      );
    case "json":
      return (
        <JsonOutputElement
          output={output as JsonInferenceOutput | undefined}
          {...props}
        />
      );
  }
}
