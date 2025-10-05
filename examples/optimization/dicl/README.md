# Example: Dynamic In-Context Learning (DICL)

This example shows how to use the dynamic in-context learning (DICL) optimization workflow to improve the performance of a variant.

For this example, we'll tackle a SMS spam classification task based on the [SMS spam dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

In the `dicl.ipynb` Jupyter notebook, we will:

1. Load the spam dataset
2. Convert it to the TensorZero format
3. Store the converted datapoints in TensorZero
4. Query the stored datapoints back
5. Launch the DICL optimization workflow
6. Compare the baseline and the DICL variants

The variant optimized with DICL materially outperforms the baseline variant.

## Setup

1. Set the `OPENAI_API_KEY` environment variable.
2. Install the Python (3.9+) dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch TensorZero:
   ```bash
   docker compose up
   ```
