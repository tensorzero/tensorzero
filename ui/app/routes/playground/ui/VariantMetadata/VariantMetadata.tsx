import type { VariantConfig } from "tensorzero-node";
import {
  Cpu,
  Weight,
  Clock,
  Users,
  Brain,
  Layers,
  Database,
  Search,
  ExternalLink,
} from "lucide-react";
import Chip from "~/components/ui/Chip";
import {
  formatNumber,
  METADATA_GRID,
  MetadataRow,
} from "./VariantMetadataUtils";

interface VariantMetadataProps {
  variant: VariantConfig;
  onOpenVariant?: (variantName: string) => void;
}

export function VariantMetadata({
  variant,
  onOpenVariant,
}: VariantMetadataProps) {
  switch (variant.type) {
    case "chat_completion":
    case "chain_of_thought": {
      return (
        <div className={METADATA_GRID}>
          <MetadataRow
            icon={<Cpu className="text-muted-foreground h-3.5 w-3.5" />}
            label="Model"
            value={variant.model}
            valueClassName="font-mono"
          />

          {variant.weight !== null && (
            <MetadataRow
              icon={<Weight className="text-muted-foreground h-3.5 w-3.5" />}
              label="Weight"
              value={formatNumber(variant.weight)}
            />
          )}
        </div>
      );
    }

    case "best_of_n_sampling": {
      return (
        <div className={METADATA_GRID}>
          <Users className="text-muted-foreground h-3.5 w-3.5" />
          <span className="text-muted-foreground text-xs">Candidates</span>
          <div className="flex flex-wrap gap-1">
            {variant.candidates.map((candidate, idx) => (
              <div
                key={idx}
                onClick={() => onOpenVariant?.(candidate)}
                className="cursor-pointer"
              >
                <Chip
                  label={candidate}
                  font="mono"
                  className="hover:bg-muted text-xs"
                  icon={<ExternalLink className="h-3 w-3" />}
                  tooltip="Open variant in view"
                />
              </div>
            ))}
          </div>

          <MetadataRow
            icon={<Brain className="text-muted-foreground h-3.5 w-3.5" />}
            label="Evaluator model"
            value={variant.evaluator.model}
            valueClassName="font-mono"
          />

          <MetadataRow
            icon={<Clock className="text-muted-foreground h-3.5 w-3.5" />}
            label="Timeout"
            value={<>{formatNumber(variant.timeout_s)}s</>}
          />

          {variant.weight !== null && (
            <MetadataRow
              icon={<Weight className="text-muted-foreground h-3.5 w-3.5" />}
              label="Weight"
              value={formatNumber(variant.weight)}
            />
          )}
        </div>
      );
    }

    case "mixture_of_n": {
      return (
        <div className={METADATA_GRID}>
          <Users className="text-muted-foreground h-3.5 w-3.5" />
          <span className="text-muted-foreground text-xs">Candidates</span>
          <div className="flex flex-wrap gap-1">
            {variant.candidates.map((candidate, idx) => (
              <div
                key={idx}
                onClick={() => onOpenVariant?.(candidate)}
                className="cursor-pointer"
              >
                <Chip
                  label={candidate}
                  font="mono"
                  className="hover:bg-muted text-xs"
                  icon={<ExternalLink className="h-3 w-3" />}
                  tooltip="Open variant in view"
                />
              </div>
            ))}
          </div>

          <MetadataRow
            icon={<Layers className="text-muted-foreground h-3.5 w-3.5" />}
            label="Fuser model"
            value={variant.fuser.model}
            valueClassName="font-mono"
          />

          <MetadataRow
            icon={<Clock className="text-muted-foreground h-3.5 w-3.5" />}
            label="Timeout"
            value={<>{formatNumber(variant.timeout_s)}s</>}
          />

          {variant.weight !== null && (
            <MetadataRow
              icon={<Weight className="text-muted-foreground h-3.5 w-3.5" />}
              label="Weight"
              value={formatNumber(variant.weight)}
            />
          )}
        </div>
      );
    }

    case "dicl": {
      return (
        <div className={METADATA_GRID}>
          <MetadataRow
            icon={<Cpu className="text-muted-foreground h-3.5 w-3.5" />}
            label="Model"
            value={variant.model}
            valueClassName="font-mono"
          />

          <MetadataRow
            icon={<Database className="text-muted-foreground h-3.5 w-3.5" />}
            label="Embedding model"
            value={variant.embedding_model}
            valueClassName="font-mono"
          />

          <MetadataRow
            icon={<Search className="text-muted-foreground h-3.5 w-3.5" />}
            label="K (retrieval)"
            value={formatNumber(variant.k)}
          />

          {variant.weight !== null && (
            <MetadataRow
              icon={<Weight className="text-muted-foreground h-3.5 w-3.5" />}
              label="Weight"
              value={formatNumber(variant.weight)}
            />
          )}
        </div>
      );
    }
  }
}
