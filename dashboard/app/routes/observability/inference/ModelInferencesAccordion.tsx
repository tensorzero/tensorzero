import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
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
      <CardHeader>
        <CardTitle className="text-xl">Model Inferences</CardTitle>
      </CardHeader>
      <CardContent>
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
