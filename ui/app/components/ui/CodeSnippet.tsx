import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Copy } from "lucide-react";
import { useRef, useState } from "react";

export interface CodeTab {
  id: string;
  label: string;
  content: string;
  language?: string;
  showLineNumbers?: boolean;
}

export interface CodeSnippetProps {
  heading?: string;
  tabs?: CodeTab[];
  label?: string;
  content?: string;
  language?: string;
  className?: string;
  children?: React.ReactNode;
  showLineNumbers?: boolean;
}

export interface CodeSnippetContentProps {
  label?: string;
  content?: string;
  language?: string;
  showLineNumbers?: boolean;
  forceShowCopyButton?: boolean;
}

export function CodeSnippetHeader({ heading }: { heading?: string }) {
  if (!heading) return null;

  return (
    <CardHeader className="pb-0">
      <CardTitle className="text-xl">{heading}</CardTitle>
    </CardHeader>
  );
}

export function CodeSnippetTabs({
  tabs,
  defaultTab,
}: {
  tabs?: CodeTab[];
  defaultTab?: string;
}) {
  if (!tabs || tabs.length === 0) return null;

  const defaultTabId = defaultTab || tabs[0].id;

  return (
    <Tabs defaultValue={defaultTabId} className="w-full">
      <TabsList className="mb-2">
        {tabs.map((tab) => (
          <TabsTrigger key={tab.id} value={tab.id}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {tabs.map((tab) => (
        <TabsContent key={tab.id} value={tab.id}>
          <CodeSnippetContent
            label={tab.label}
            content={tab.content}
            language={tab.language}
            showLineNumbers={tab.showLineNumbers}
          />
        </TabsContent>
      ))}
    </Tabs>
  );
}

export function CodeSnippetContent({
  label,
  content,
  showLineNumbers = false,
  forceShowCopyButton = false,
}: CodeSnippetContentProps) {
  if (!content) return null;
  const [error, setError] = useState<string | null>(null);

  const isSecureContext =
    typeof window !== "undefined" && window.isSecureContext;
  const isClipboard = typeof navigator !== "undefined" && !!navigator.clipboard;
  const shouldShowCopyButton =
    (isSecureContext && isClipboard) || forceShowCopyButton;
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to copy. Please copy manually.");
      if (textAreaRef.current) {
        textAreaRef.current.select();
      }
    }
  };

  const lines = content.split("\n");

  return (
    <div className="relative">
      {label && <Badge className="mb-2">{label}</Badge>}

      <div className="relative rounded-lg bg-background-primary">
        {shouldShowCopyButton && (
          <Button
            variant="outline"
            size="icon"
            onClick={handleCopy}
            className="absolute right-2 top-2 z-10 h-7 w-7 p-0 shadow-none"
            aria-label="Copy code"
          >
            <Copy className="h-4 w-4" />
          </Button>
        )}

        <textarea
          ref={textAreaRef}
          value={content}
          readOnly
          className="sr-only"
          aria-hidden="true"
        />

        {error && (
          <div className="absolute right-2 top-10 z-10 mt-2 rounded bg-red-100 p-2 text-xs text-red-800">
            {error}
          </div>
        )}

        <div className="relative overflow-x-auto">
          <div className="flex">
            {showLineNumbers && (
              <div className="flex-shrink-0 select-none pb-4 pl-4 pr-3 pt-4 text-right font-mono text-gray-400">
                {lines.map((_, i) => (
                  <div key={i} className="h-[1.5rem] text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <pre
              className={`flex-1 overflow-x-auto whitespace-pre break-words p-4`}
            >
              <code className="font-mono text-sm leading-6 text-foreground-secondary">
                {lines.map((line, i) => (
                  <div key={i} className="h-[1.5rem]">
                    {line || " "}
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

export function CodeSnippet({
  heading,
  tabs,
  label,
  content,
  language,
  className = "",
  children,
  showLineNumbers,
}: CodeSnippetProps) {
  if (children) {
    return <Card className={className}>{children}</Card>;
  }

  return (
    <Card className={className}>
      <CodeSnippetHeader heading={heading} />

      <CardContent className="space-y-4 pt-6">
        {tabs && tabs.length > 0 ? (
          <CodeSnippetTabs tabs={tabs} />
        ) : (
          <CodeSnippetContent
            label={label}
            content={content}
            language={language}
            showLineNumbers={showLineNumbers}
          />
        )}
      </CardContent>
    </Card>
  );
}
