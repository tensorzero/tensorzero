import { cn } from "~/utils/common";

/** Direction the gradient fades toward */
export const FadeDirection = {
  /** Fades from solid at top to transparent at bottom */
  Top: "top",
  /** Fades from solid at bottom to transparent at top */
  Bottom: "bottom",
} as const;

export type FadeDirection = (typeof FadeDirection)[keyof typeof FadeDirection];

/** Common surface colors for backgrounds and gradients */
export const SurfaceColor = {
  Primary: "primary",
  Secondary: "secondary",
  Tertiary: "tertiary",
} as const;

export type SurfaceColor = (typeof SurfaceColor)[keyof typeof SurfaceColor];

const surfaceColorClasses: Record<SurfaceColor, string> = {
  [SurfaceColor.Primary]: "from-bg-primary",
  [SurfaceColor.Secondary]: "from-bg-secondary",
  [SurfaceColor.Tertiary]: "from-bg-tertiary",
};

type FadeGradientProps = {
  direction: FadeDirection;
  visible: boolean;
  color?: SurfaceColor;
  className?: string;
};

export function FadeGradient({
  direction,
  visible,
  color = SurfaceColor.Secondary,
  className,
}: FadeGradientProps) {
  return (
    <div
      className={cn(
        "h-16",
        surfaceColorClasses[color],
        "to-transparent",
        direction === FadeDirection.Top
          ? "bg-gradient-to-b"
          : "bg-gradient-to-t",
        "transition-opacity duration-75",
        visible ? "opacity-100" : "opacity-0",
        className,
      )}
    />
  );
}
