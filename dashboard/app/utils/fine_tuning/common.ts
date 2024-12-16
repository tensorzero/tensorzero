import { ParsedInferenceRow } from "../clickhouse";

export function splitValidationData(
  inferences: ParsedInferenceRow[],
  validationSplit: number,
) {
  const splitIndex =
    validationSplit > 0
      ? Math.floor(inferences.length * (1 - validationSplit / 100))
      : inferences.length;

  const trainInferences = inferences.slice(0, splitIndex);
  const valInferences = validationSplit > 0 ? inferences.slice(splitIndex) : [];

  return {
    trainInferences,
    valInferences,
  };
}
