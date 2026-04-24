export function createNewConfigPath(existingPaths: Iterable<string>): string {
  const existing = new Set(existingPaths);
  let index = 1;

  while (true) {
    const candidate = `new-template-${index}`;
    if (!existing.has(candidate)) {
      return candidate;
    }
    index += 1;
  }
}
