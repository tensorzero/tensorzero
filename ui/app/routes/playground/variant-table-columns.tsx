import { type ColumnDef } from "@tanstack/react-table";
import { type VariantConfig } from "~/utils/config/variant";
import { VariantTypeBadge } from "~/components/ui/variant-type-badge";
import { Badge } from "~/components/ui/badge";

export type VariantTableRow = {
  id: string;
  name: string;
  config: VariantConfig;
};

export const createVariantColumns = (): ColumnDef<VariantTableRow>[] => [
  {
    id: "name",
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => (
      <div className="font-medium">{row.getValue("name")}</div>
    ),
  },
  {
    id: "type",
    accessorFn: (row) => row.config.type,
    header: "Type",
    cell: ({ row }) => (
      <VariantTypeBadge type={row.original.config.type} />
    ),
  },
  {
    id: "weight",
    accessorFn: (row) => row.config.weight,
    header: "Weight",
    cell: ({ row }) => (
      <div className="text-right font-mono">
        {row.original.config.weight}
      </div>
    ),
  },
  {
    id: "model",
    accessorFn: (row) => {
      const config = row.config;
      if ("model" in config) {
        return config.model;
      }
      return null;
    },
    header: "Model",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("model" in config && config.model) {
        return (
          <Badge variant="outline" className="font-mono text-xs">
            {config.model}
          </Badge>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "temperature",
    accessorFn: (row) => {
      const config = row.config;
      if ("temperature" in config) {
        return config.temperature;
      }
      return null;
    },
    header: "Temperature",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("temperature" in config && config.temperature !== undefined) {
        return (
          <div className="text-right font-mono">
            {config.temperature}
          </div>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "max_tokens",
    accessorFn: (row) => {
      const config = row.config;
      if ("max_tokens" in config) {
        return config.max_tokens;
      }
      return null;
    },
    header: "Max Tokens",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("max_tokens" in config && config.max_tokens !== undefined) {
        return (
          <div className="text-right font-mono">
            {config.max_tokens.toLocaleString()}
          </div>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "timeout_s",
    accessorFn: (row) => {
      const config = row.config;
      if ("timeout_s" in config) {
        return config.timeout_s;
      }
      return null;
    },
    header: "Timeout (s)",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("timeout_s" in config && config.timeout_s !== undefined) {
        return (
          <div className="text-right font-mono">
            {config.timeout_s}s
          </div>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "candidates",
    accessorFn: (row) => {
      const config = row.config;
      if ("candidates" in config) {
        return config.candidates?.length || 0;
      }
      return null;
    },
    header: "Candidates",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("candidates" in config && config.candidates) {
        return (
          <Badge variant="secondary">
            {config.candidates.length} candidates
          </Badge>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "embedding_model",
    accessorFn: (row) => {
      const config = row.config;
      if ("embedding_model" in config) {
        return config.embedding_model;
      }
      return null;
    },
    header: "Embedding Model",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("embedding_model" in config && config.embedding_model) {
        return (
          <Badge variant="outline" className="font-mono text-xs">
            {config.embedding_model}
          </Badge>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "k",
    accessorFn: (row) => {
      const config = row.config;
      if ("k" in config) {
        return config.k;
      }
      return null;
    },
    header: "K",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("k" in config && config.k !== undefined) {
        return (
          <div className="text-right font-mono">
            {config.k}
          </div>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
  {
    id: "json_mode",
    accessorFn: (row) => {
      const config = row.config;
      if ("json_mode" in config) {
        return config.json_mode;
      }
      return null;
    },
    header: "JSON Mode",
    cell: ({ row }) => {
      const config = row.original.config;
      if ("json_mode" in config && config.json_mode) {
        const variant = config.json_mode === "on" ? "default" : "secondary";
        return (
          <Badge variant={variant} className="text-xs">
            {config.json_mode}
          </Badge>
        );
      }
      return <span className="text-muted-foreground">—</span>;
    },
  },
];