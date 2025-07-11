import { motion, AnimatePresence } from "motion/react";

/** Animated fade transition wrapper */
export default function FadeTransition({
  children,
  stateKey,
  className,
}: React.PropsWithChildren<{
  stateKey: string;
  className?: string;
}>) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={stateKey}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className={className}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
