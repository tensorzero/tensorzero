import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Card, CardContent } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { ModelInferenceItem } from "./ModelInferenceItem";

interface ModelInferencesAccordionProps {
  modelInferences: ParsedModelInferenceRow[];
}

export function ModelInferencesAccordion({
  modelInferences,
}: ModelInferencesAccordionProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <Accordion type="single" collapsible className="w-full">
          {modelInferences.map((inference, index) => (
            <AccordionItem value={`item-${index}`} key={inference.id}>
              <AccordionTrigger>
                <Code>{inference.id}</Code>
              </AccordionTrigger>
              <AccordionContent>
                <ModelInferenceItem inference={inference} />
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  );
}
