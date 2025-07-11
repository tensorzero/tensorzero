import clsx from "clsx";
import { ChevronDownIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "~/components/ui/collapsible";
import { cn } from "~/utils/common";

// TODO Fade in *after* layout transition

const AnimatedCollapsible: React.FC<
  React.PropsWithChildren<{
    label: string;
    isOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
    className?: string;
  }>
> = ({ label, isOpen, onOpenChange, className, children }) => {
  return (
    <Collapsible
      open={isOpen}
      onOpenChange={onOpenChange}
      className={cn("flex flex-col gap-1", className)}
    >
      <CollapsibleTrigger
        aria-label={`${isOpen ? "Collapse" : "Expand"} ${label} section`}
        className="group flex cursor-pointer items-center gap-2 text-sm font-normal text-gray-600 transition-colors hover:text-gray-900 focus:outline-none"
      >
        <span>{label}</span>
        <ChevronDownIcon
          className={clsx(
            "h-3 w-3 text-gray-400 transition-transform duration-200",
            isOpen && "rotate-180",
          )}
        />
      </CollapsibleTrigger>
      <AnimatePresence>
        {isOpen && (
          <CollapsibleContent forceMount asChild>
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{
                duration: 0.15,
                ease: "easeInOut",
              }}
            >
              {children}
            </motion.div>
          </CollapsibleContent>
        )}
      </AnimatePresence>
    </Collapsible>
  );
};

export default AnimatedCollapsible;
