import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";

interface Message {
  role: string;
  content: string;
}

interface FirstExampleProps {
  messages: Message[];
}

export function FirstExample({ messages }: FirstExampleProps) {
  if (!messages || messages.length === 0) return null;

  return (
    <Accordion type="single" collapsible className="w-full rounded-md border">
      <AccordionItem value="first-example" className="border-none">
        <AccordionTrigger className="px-4">First Example</AccordionTrigger>
        <AccordionContent className="space-y-4 px-4">
          <div className="space-y-2">
            {messages.map((message, index) => (
              <div key={index} className="space-y-1">
                <div className="text-sm text-muted-foreground">
                  {message.role}:
                </div>
                <code className="block whitespace-pre-wrap rounded-lg bg-muted p-3 text-sm">
                  {message.content}
                </code>
              </div>
            ))}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
