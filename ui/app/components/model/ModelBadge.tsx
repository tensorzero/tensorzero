import { Badge } from "~/components/ui/badge";
import type { ProviderConfig } from "tensorzero-node";
import { formatProvider } from "~/utils/providers";

interface ModelBadgeProps {
  provider: ProviderConfig["type"];
}

export function ModelBadge({ provider }: ModelBadgeProps) {
  const providerInfo = formatProvider(provider);

  return (
    <Badge
      className={`${providerInfo.className} hover:${providerInfo.className}`}
    >
      {providerInfo.name}
    </Badge>
  );
}
