// const useDatapoint = (dataset_name: string, id: string) => {
//   // await loader();
//   // return useRouteLoaderData("datasets/$dataset_name/datapoint/$id");
//   // const datapoint = useLoaderData();
//   // return datapoint;

//   const fetcher = useFetcher();
//   fetcher.load(`/datasets/${dataset_name}/datapoint/${id}`);
//   return fetcher.data?.datapoint;
// };

// const DatapointsPanel: React.FC<React.PropsWithChildren<{}>> = ({
//   children,
// }) => {
//   const DATAPOINT_ID = "0197b21f-5e78-7a92-931a-20ad51930336"; // For `tensorzero::llm_judge::entity_extraction::count_sports`

//   const datapoint = useDatapoint(
//     "tensorzero::llm_judge::entity_extraction::count_sports",
//     DATAPOINT_ID,
//   );

//   // TODO Can I query a datapoint here?

//   return (
//     <div className="row-start-1 -row-end-2 grid grid-cols-1 grid-rows-subgrid gap-y-4">
//       {children}
//     </div>
//   );
// };
