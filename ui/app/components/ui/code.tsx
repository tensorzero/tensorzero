import React from "react";

interface CodeProps extends React.HTMLAttributes<HTMLElement> {
  children: React.ReactNode;
}

export function Code({ children, className, ...props }: CodeProps) {
  return (
    <code
      className={`rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold ${className}`}
      {...props}
    >
      {children}
    </code>
  );
}
