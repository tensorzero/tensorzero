import { ProviderBadge } from "~/components/ui/provider-badge";
import type { ProviderType } from "~/utils/config/models";

type ModelBadgesProps = {
  provider: ProviderType;
  modelName?: string;
  showModelName?: boolean;
  compact?: boolean;
  className?: string;
};

export function ModelBadges({ 
  provider, 
  modelName, 
  showModelName = false, 
  compact = false,
  className 
}: ModelBadgesProps) {
  // Create a minimal provider config for the ProviderBadge
  const providerConfig = {
    type: provider,
    ...(modelName && { model_name: modelName })
  } as const;

  return (
    <ProviderBadge 
      provider={providerConfig as any}
      showModelName={showModelName}
      compact={compact}
      className={className}
    />
  );
}
