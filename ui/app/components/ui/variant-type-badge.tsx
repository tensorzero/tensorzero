import { Badge } from "./badge";
import { type VariantType } from "~/utils/config/variant";

interface VariantTypeBadgeProps {
  type: VariantType;
}

const VARIANT_TYPE_CONFIG = {
  chat_completion: {
    label: "Chat Completion",
    variant: "default" as const,
    className: "bg-blue-100 text-blue-800 hover:bg-blue-100",
  },
  experimental_best_of_n_sampling: {
    label: "Best of N",
    variant: "secondary" as const,
    className: "bg-purple-100 text-purple-800 hover:bg-purple-100",
  },
  experimental_dynamic_in_context_learning: {
    label: "DICL",
    variant: "secondary" as const,
    className: "bg-orange-100 text-orange-800 hover:bg-orange-100",
  },
  experimental_mixture_of_n: {
    label: "Mixture of N",
    variant: "secondary" as const,
    className: "bg-green-100 text-green-800 hover:bg-green-100",
  },
  experimental_chain_of_thought: {
    label: "Chain of Thought",
    variant: "secondary" as const,
    className: "bg-cyan-100 text-cyan-800 hover:bg-cyan-100",
  },
} as const;

export function VariantTypeBadge({ type }: VariantTypeBadgeProps) {
  const config = VARIANT_TYPE_CONFIG[type];
  
  return (
    <Badge 
      variant={config.variant}
      className={config.className}
    >
      {config.label}
    </Badge>
  );
}