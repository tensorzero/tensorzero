type ComboboxHintProps = {
  children: React.ReactNode;
};

export function ComboboxHint({ children }: ComboboxHintProps) {
  return (
    <div className="text-muted-foreground border-t px-3 py-2 text-xs">
      {children}
    </div>
  );
}
