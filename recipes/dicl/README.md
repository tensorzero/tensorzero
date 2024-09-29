# Dynamic In-Context Learning

LLMs are excellent few-shot learners.
As an intermediate point between instruction based prompting (zero-shot LLM generation) and a fine-tune, in-context learning with examples is a convenient and highly effective way to improve performance.
In particular, for a particular input, it is often a good idea to find the most similar set of examples of good behavior and use them as the example for few-shot learning in context.

As TensorZero is designed to store inferences and feedback in structured format, it is easy to query a dataframe of examples that were successful and then use them to do in-context learning with a new input.

In this notebook, we give an example of how to do this for inferences which have scored well according to a boolean or float metric, or for inferences which have demonstrations of ideal behavior.
In the case of float metrics, we offer the option to choose a cutoff for the metric score which qualifies an inference as a success and worthy of inclusion in the dynamic in-context learning examples.
In the case of demonstrations, we assume all demonstrations are good enough.
We expose all these settings in the front of `dicl.ipynb`.
You might also want to use your own strategies for choosing the examples.
This is fine and even encouraged.
As long as you store the inputs in the correct format in the DynamicInContextLearningExample table, you can use any strategy you like for choosing the examples.
Feel free to edit this notebook to use your strategy of choice.
