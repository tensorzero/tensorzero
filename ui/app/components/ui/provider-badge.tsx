// import { Badge } from "./badge";
// import { cn } from "~/utils/common";
// import type { ProviderConfig, ProviderType } from "~/utils/config/models";

// // Lobe Icons imports - using specific paths to avoid module resolution issues
// import OpenAI from "@lobehub/icons/es/OpenAI";
// import Anthropic from "@lobehub/icons/es/Anthropic";
// import Google from "@lobehub/icons/es/Google";
// import Aws from "@lobehub/icons/es/Aws";
// import Azure from "@lobehub/icons/es/Azure";
// import Mistral from "@lobehub/icons/es/Mistral";
// import Groq from "@lobehub/icons/es/Groq";
// import DeepSeek from "@lobehub/icons/es/DeepSeek";
// import Fireworks from "@lobehub/icons/es/Fireworks";
// import Together from "@lobehub/icons/es/Together";
// import XAI from "@lobehub/icons/es/XAI";
// import Hyperbolic from "@lobehub/icons/es/Hyperbolic";

// interface ProviderBadgeProps {
//   provider: ProviderConfig;
//   showModelName?: boolean;
//   compact?: boolean;
//   className?: string;
// }

// interface ProviderInfo {
//   name: string;
//   icon: React.ComponentType<{ size?: number; className?: string }>;
//   iconColor: string;
//   borderColor: string;
//   textColor: string;
//   bgColor: string;
// }

// function getProviderInfo(providerType: ProviderType): ProviderInfo {
//   switch (providerType) {
//     case "openai":
//       return {
//         name: "OpenAI",
//         icon: OpenAI,
//         iconColor: "#412991",
//         borderColor: "border-[#412991]",
//         textColor: "text-[#412991]",
//         bgColor: "bg-[#412991]/5",
//       };

//     case "anthropic":
//       return {
//         name: "Anthropic",
//         icon: Anthropic,
//         iconColor: "#D4A574",
//         borderColor: "border-[#D4A574]",
//         textColor: "text-[#D4A574]",
//         bgColor: "bg-[#D4A574]/5",
//       };

//     case "google_ai_studio_gemini":
//       return {
//         name: "Google AI Studio",
//         icon: Google,
//         iconColor: "#4285F4",
//         borderColor: "border-[#4285F4]",
//         textColor: "text-[#4285F4]",
//         bgColor: "bg-[#4285F4]/5",
//       };

//     case "gcp_vertex_gemini":
//       return {
//         name: "Vertex AI (Gemini)",
//         icon: Google,
//         iconColor: "#4285F4",
//         borderColor: "border-[#4285F4]",
//         textColor: "text-[#4285F4]",
//         bgColor: "bg-[#4285F4]/5",
//       };

//     case "gcp_vertex_anthropic":
//       return {
//         name: "Vertex AI (Anthropic)",
//         icon: Google,
//         iconColor: "#4285F4",
//         borderColor: "border-[#4285F4]",
//         textColor: "text-[#4285F4]",
//         bgColor: "bg-[#4285F4]/5",
//       };

//     case "aws_bedrock":
//       return {
//         name: "AWS Bedrock",
//         icon: Aws,
//         iconColor: "#FF9900",
//         borderColor: "border-[#FF9900]",
//         textColor: "text-[#FF9900]",
//         bgColor: "bg-[#FF9900]/5",
//       };

//     case "aws_sagemaker":
//       return {
//         name: "AWS SageMaker",
//         icon: Aws,
//         iconColor: "#FF9900",
//         borderColor: "border-[#FF9900]",
//         textColor: "text-[#FF9900]",
//         bgColor: "bg-[#FF9900]/5",
//       };

//     case "azure":
//       return {
//         name: "Azure OpenAI",
//         icon: Azure,
//         iconColor: "#0078D4",
//         borderColor: "border-[#0078D4]",
//         textColor: "text-[#0078D4]",
//         bgColor: "bg-[#0078D4]/5",
//       };

//     case "mistral":
//       return {
//         name: "Mistral AI",
//         icon: Mistral,
//         iconColor: "#FF7000",
//         borderColor: "border-[#FF7000]",
//         textColor: "text-[#FF7000]",
//         bgColor: "bg-[#FF7000]/5",
//       };

//     case "groq":
//       return {
//         name: "Groq",
//         icon: Groq,
//         iconColor: "#F55036",
//         borderColor: "border-[#F55036]",
//         textColor: "text-[#F55036]",
//         bgColor: "bg-[#F55036]/5",
//       };

//     case "deepseek":
//       return {
//         name: "DeepSeek",
//         icon: DeepSeek,
//         iconColor: "#1C64F2",
//         borderColor: "border-[#1C64F2]",
//         textColor: "text-[#1C64F2]",
//         bgColor: "bg-[#1C64F2]/5",
//       };

//     case "fireworks":
//       return {
//         name: "Fireworks AI",
//         icon: Fireworks,
//         iconColor: "#FF6B35",
//         borderColor: "border-[#FF6B35]",
//         textColor: "text-[#FF6B35]",
//         bgColor: "bg-[#FF6B35]/5",
//       };

//     case "together":
//       return {
//         name: "Together AI",
//         icon: Together,
//         iconColor: "#6366F1",
//         borderColor: "border-[#6366F1]",
//         textColor: "text-[#6366F1]",
//         bgColor: "bg-[#6366F1]/5",
//       };

//     case "xai":
//       return {
//         name: "xAI",
//         icon: XAI,
//         iconColor: "#000000",
//         borderColor: "border-gray-800",
//         textColor: "text-gray-800",
//         bgColor: "bg-gray-50",
//       };

//     case "hyperbolic":
//       return {
//         name: "Hyperbolic",
//         icon: Hyperbolic,
//         iconColor: "#EC4899",
//         borderColor: "border-[#EC4899]",
//         textColor: "text-[#EC4899]",
//         bgColor: "bg-[#EC4899]/5",
//       };

//     case "openrouter":
//       return {
//         name: "OpenRouter",
//         icon: OpenAI, // Using OpenAI icon as fallback
//         iconColor: "#6366F1",
//         borderColor: "border-indigo-500",
//         textColor: "text-indigo-600",
//         bgColor: "bg-indigo-50",
//       };

//     case "vllm":
//       return {
//         name: "vLLM",
//         icon: () => (
//           <div className="flex h-4 w-4 items-center justify-center rounded bg-current text-xs font-bold text-white">
//             V
//           </div>
//         ),
//         iconColor: "#06B6D4",
//         borderColor: "border-cyan-500",
//         textColor: "text-cyan-600",
//         bgColor: "bg-cyan-50",
//       };

//     case "sglang":
//       return {
//         name: "SGLang",
//         icon: () => (
//           <div className="flex h-4 w-4 items-center justify-center rounded bg-current text-xs font-bold text-white">
//             S
//           </div>
//         ),
//         iconColor: "#10B981",
//         borderColor: "border-emerald-500",
//         textColor: "text-emerald-600",
//         bgColor: "bg-emerald-50",
//       };

//     case "tgi":
//       return {
//         name: "Text Generation Inference",
//         icon: () => (
//           <div className="flex h-4 w-4 items-center justify-center rounded bg-current text-xs font-bold text-white">
//             T
//           </div>
//         ),
//         iconColor: "#10B981",
//         borderColor: "border-emerald-500",
//         textColor: "text-emerald-600",
//         bgColor: "bg-emerald-50",
//       };

//     case "dummy":
//       return {
//         name: "Dummy",
//         icon: () => (
//           <div className="flex h-4 w-4 items-center justify-center rounded bg-current text-xs font-bold text-white">
//             ?
//           </div>
//         ),
//         iconColor: "#6B7280",
//         borderColor: "border-gray-400",
//         textColor: "text-gray-600",
//         bgColor: "bg-gray-50",
//       };

//     default:
//       return {
//         name: "Unknown Provider",
//         icon: () => (
//           <div className="flex h-4 w-4 items-center justify-center rounded bg-current text-xs font-bold text-white">
//             ?
//           </div>
//         ),
//         iconColor: "#6B7280",
//         borderColor: "border-gray-400",
//         textColor: "text-gray-600",
//         bgColor: "bg-gray-50",
//       };
//   }
// }

// function getModelName(provider: ProviderConfig): string {
//   if ("model_name" in provider && provider.model_name) {
//     return provider.model_name;
//   }
//   if ("model_id" in provider && provider.model_id) {
//     return provider.model_id;
//   }
//   if ("endpoint_name" in provider && provider.endpoint_name) {
//     return provider.endpoint_name;
//   }
//   return "";
// }

// export function ProviderBadge({
//   provider,
//   showModelName = false,
//   compact = false,
//   className,
// }: ProviderBadgeProps) {
//   const info = getProviderInfo(provider.type);
//   const modelName = getModelName(provider);
//   const IconComponent = info.icon;

//   const displayText = compact
//     ? info.name.split(" ")[0] // First word only in compact mode
//     : showModelName && modelName
//       ? `${info.name} · ${modelName}`
//       : info.name;

//   return (
//     <Badge
//       variant="outline"
//       className={cn(
//         "inline-flex items-center gap-1.5 font-medium",
//         info.borderColor,
//         info.textColor,
//         info.bgColor,
//         "hover:bg-opacity-10",
//         compact ? "px-2 py-0.5 text-xs" : "px-2.5 py-0.5 text-xs",
//         className,
//       )}
//     >
//       <IconComponent size={compact ? 12 : 14} className="flex-shrink-0" />
//       <span className={cn(compact && "sr-only sm:not-sr-only")}>
//         {displayText}
//       </span>
//     </Badge>
//   );
// }

// export { type ProviderBadgeProps };
