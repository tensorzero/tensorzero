export const mockDatasets = [
  {
    name: "test_dataset",
    count: 3250,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
  },
  {
    name: "evaluation_set",
    count: 850,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
  },
  {
    name: "training_data_v2",
    count: 42100,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(),
  },
  {
    name: "validation_set",
    count: 1200,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
  },
  {
    name: "production_data",
    count: 15420,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
  },
  {
    name: "long_dataset_name_that_should_still_be_displayed_gracefully_somehow",
    count: 1,
    lastUpdated: new Date(Date.now()).toISOString(),
  },
];

export function createManyDatasets(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    name: `test_dataset_${i + 1}`,
    count: i + 1,
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * (i + 1)).toISOString(),
  }));
}
