.. acies-vehicle-classifier documentation master file, created by
   sphinx-quickstart on Sun Aug 24 23:04:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Acies Vehicle Classifier
========================

Introduction
------------

A real-time vehicle classification system that processes multimodal sensor data
(seismic and acoustic) to identify vehicles using foundation models. The system
supports state synchronization, temporal ensemble voting, and distributed
processing through the AciesOS middleware.

The :class:`~acies.vehicle_classifier.base.Classifier` class implements core
infrastructure including: stream data buffering to accumulate sensor readings
over configurable time windows, energy-based filtering to reject low-energy
noise from seismic and acoustic sensors, temporal ensemble voting that combines
multiple classification results to improve accuracy over time, automated
scheduling of inference cycles and message handling, multimodal data fusion
supporting both seismic and acoustic modalities, and comprehensive logging of
classification results with confidence scores, ground truth comparison, and
latency metrics.

Concrete classifiers must implement two essential methods:
:meth:`~acies.vehicle_classifier.base.Classifier.load_model` which loads the
classification model from a configuration file and sets the supported
modalities, and :meth:`~acies.vehicle_classifier.base.Classifier.infer` which
performs inference on buffered sensor data and returns classification
probabilities. The base classifier calls
:meth:`~acies.vehicle_classifier.base.Classifier.load_model` during
initialization and periodically invokes
:meth:`~acies.vehicle_classifier.base.Classifier.infer` with accumulated
multimodal sensor samples when sufficient data is available.

The :class:`~acies.vehicle_classifier.vfm.VibroFM` class provides a concrete
implementation of the abstract
:meth:`~acies.vehicle_classifier.base.Classifier.load_model` and
:meth:`~acies.vehicle_classifier.base.Classifier.infer` methods, building on our
foundation model work ([Kimura2024]_, [Liu2023]_, [Kara2024]_). For details on
the expected input format, refer to the
:meth:`~acies.vehicle_classifier.base.Classifier.infer` docstring. Pre-trained
and fine-tuned model weights from [Kimura2024]_ and [Kara2024]_, compatible with
VibroFM, are available on the `GitHub release page
<https://github.com/acies-os/vehicle-classifier/releases>`_.

Implement A New Classifier
--------------------------

You can clone the repository wherever you prefer — in this guide, we'll use ``/ws/acies`` as an example.
Assume the repository is set up as follows:

.. code-block:: bash

    $ cd /ws/acies/
    /ws/acies$ git clone git@github.com:acies-os/vehicle-classifier.git
    /ws/acies$ cd vehicle-classifier
    /ws/acies/vehicle-classifier$ uv sync

To create a new classifier, follow these steps:

1. **Copy the example template** to create your new classifier file:

.. code-block:: bash

    vehicle-classifier$ cp src/acies/vehicle_classifier/example.py src/acies/vehicle_classifier/my_classifier.py

2. **Implement the required methods** in your new classifier file:

   - ``load_model()``: Load your model and set ``self.modalities``
   - ``infer()``: Perform inference on sensor data and return class probabilities

3. **Add an entry** to ``pyproject.toml`` under ``[project.scripts]``:

.. code-block:: toml

    [project.scripts]
    my-clf = "acies.vehicle_classifier.my_classifier:main"

4. **Run your new classifier** using uv:

.. code-block:: bash

    vehicle-classifier$ uv run my-clf --weight /path/to/config --namespace ns1 --proc-name clf1

API
---

.. toctree::
   :maxdepth: 3
   :caption: Content

   api

References
----------

.. [Kimura2024] Kimura, Tomoyoshi, Jinyang Li, Tianshi Wang, Yizhuo Chen, Ruijie
    Wang, Denizhan Kara, Maggie Wigness et al. "Vibrofm: Towards micro
    foundation models for robust multimodal iot sensing." In *2024 IEEE 21st
    International Conference on Mobile Ad-Hoc and Smart Systems (MASS)*, pp.
    10-18. IEEE, 2024.

.. [Liu2023] Liu, Shengzhong, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang,
    Jinyang Li, Suhas Diggavi, Mani Srivastava, and Tarek Abdelzaher. "Focal:
    Contrastive learning for multimodal time-series sensing signals in
    factorized orthogonal latent space." *Advances in Neural Information
    Processing Systems* 36 (2023): 47309-47338.

.. [Kara2024] Kara, Denizhan, Tomoyoshi Kimura, Yatong Chen, Jinyang Li, Ruijie
    Wang, Yizhuo Chen, Tianshi Wang, Shengzhong Liu, and Tarek Abdelzaher.
    "Phymask: An adaptive masking paradigm for efficient self-supervised
    learning in iot." In *Proceedings of the 22nd ACM conference on embedded
    networked sensor systems*, pp. 97-111. 2024.
