"""Example classifier implementation.

This module provides a template for implementing a custom vehicle classifier
that extends the base Classifier class. Replace the placeholder implementations
with your actual model loading and inference logic.
"""

import logging
from pathlib import Path

import click
import numpy as np

from acies.core import common_options, get_zconf, init_logger
from acies.vehicle_classifier.base import Classifier

logger = logging.getLogger('acies.infer')


class ExampleClassifier(Classifier):
    """Example vehicle classifier implementation.

    This is a template class that demonstrates how to implement a concrete
    classifier by extending the base Classifier class. You must implement
    the load_model and infer methods.
    """

    def __init__(self, classifier_config_file, *args, **kwargs):
        """Initialize the ExampleClassifier.

        Parameters
        ----------
        classifier_config_file : :class:`pathlib.Path`
            The path to the classifier configuration file.
        """
        super().__init__(classifier_config_file, *args, **kwargs)

    def load_model(self, classifier_config_file: Path):
        """Load the model for inference.

        This method must be implemented to load your classification model
        from the provided configuration file and set the supported modalities.

        Parameters
        ----------
        classifier_config_file : :class:`pathlib.Path`
            The path to the classifier configuration file.

        Returns
        -------
        Any
            The loaded model object (can be any type depending on your implementation).
        """
        # TODO: Implement your model loading logic here
        # Example:
        # model = load_your_model(classifier_config_file)

        # Set the modalities that your model supports
        # Common modalities are 'geo' (seismic) and 'mic' (acoustic)
        self.modalities = ['geo', 'mic']  # Replace with your supported modalities

        logger.info(f'Loaded example model from {classifier_config_file}')
        logger.info(f'Supported modalities: {self.modalities}')

        # Return your loaded model
        return None  # Replace with your actual model

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        """Run inference on the provided samples.

        This method must be implemented to perform classification on the
        buffered sensor data and return classification probabilities.

        Parameters
        ----------
        samples : dict[str, dict[int, numpy.ndarray]]
            Nested mapping of time-series data.

            - Keys (``str``): Topic name, such as ``rs1/mic`` or ``rs1/geo``.
            - Values (``dict``): Data received from that topic.
                - Keys (``int``): Unix timestamp in seconds.
                - Values (``numpy.ndarray``): One-dimensional arrays containing
                  the raw samples recorded during that second.

        Returns
        -------
        dict[int, float]
            Classification results as a dictionary mapping class indices to
            confidence scores/probabilities.
        """
        # TODO: Implement your inference logic here

        # Example preprocessing: concatenate time-series data for each modality
        arrays = {k: self.concat(v) for k, v in samples.items()}
        arrays = {k.split('/')[-1]: v for k, v in arrays.items()}

        logger.debug(f'Processing samples with shapes: {[f"{k}: {v.shape}" for k, v in arrays.items()]}')

        # TODO: Preprocess your data as needed for your model
        # Example:
        # processed_data = preprocess(arrays)

        # TODO: Run inference with your model
        # Example:
        # predictions = self.model(processed_data)

        # TODO: Return classification results
        # The result should be a dictionary mapping class indices to confidence scores
        # Example for 4 classes with dummy predictions:
        result = {
            0: 0.25,  # polaris
            1: 0.25,  # warthog
            2: 0.25,  # truck
            3: 0.25,  # husky
        }

        logger.debug(f'Classification result: {result}')
        return result


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=click.Path(exists=True))
@click.option('--sync-interval', help='Sync interval in seconds', type=int, default=1)
@click.option('--feature-twin', help='Enable digital twin features', is_flag=True, default=False)
@click.option('--twin-model', help='Model used in the digital twin', type=str, default='multimodal')
@click.option('--twin-buff-len', help='Buffer length in the digital twin', type=int, default=2)
@click.option('--heartbeat-interval-s', help='Heartbeat interval in seconds', type=int, default=5)
@click.option('--my-additional-option', help='Example of additional custom option', type=str, default='default_value')
@click.argument('my-additional-argument', help='Example of additional custom argument', required=False)
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    deactivated,
    weight,
    sync_interval,
    feature_twin,
    twin_model,
    twin_buff_len,
    heartbeat_interval_s,
    my_additional_option,
    my_additional_argument,
):
    """Main entry point for the example classifier."""

    log_file = f'{namespace.replace("/", "_")}_{proc_name.replace("/", "_")}.log'
    init_logger(log_file, name='acies')
    z_conf = get_zconf(mode, connect, listen)

    # Example usage of additional custom parameters
    logger.info(f'Custom option value: {my_additional_option}')
    if my_additional_argument:
        logger.info(f'Custom argument value: {my_additional_argument}')

    # initialize the classifier
    clf = ExampleClassifier(
        conf=z_conf,
        twin_model=twin_model,
        twin_buff_len=twin_buff_len,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        deactivated=deactivated,
        classifier_config_file=weight,
        sync_interval=sync_interval,
        feature_twin=feature_twin,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start the classifier
    clf.start()


if __name__ == '__main__':
    main()
