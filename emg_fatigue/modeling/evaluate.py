import tensorflow as tf
from tabulate import tabulate


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
    table_format: str = "grid",
) -> dict:
    """
    Evaluates a Keras model on the test dataset and prints the results in a pretty table.

    Args:
        model: A compiled TensorFlow Keras model.
        test_ds: A tf.data.Dataset representing the test set.
        model_name: A string name for the model to be printed in the table.
        table_format: Format to use with `tabulate`, e.g., "grid", "fancy_grid", "plain".

    Returns:
        A dictionary mapping metric names to their evaluation results.
    """
    # Perform evaluation
    results = model.evaluate(test_ds, return_dict=True, verbose=0)

    # Prepare data for tabulate
    headers = ["Model"] + list(results.keys())
    row = [model_name] + [f"{v:.4f}" for v in results.values()]
    table = [row]

    # Print table
    print(tabulate(table, headers=headers, tablefmt=table_format))

    return results
