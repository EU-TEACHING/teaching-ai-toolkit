# teaching-ai-toolkit
AI-Toolkit collecting and implementing the AI modules for a TEACHING application.

## Description
A TEACHING app defines a computational graph as a docker.compose application.
Such graph is composed of several nodes that act as *Producers*, *Consumers* or both.
The communication (exchange of JSON based data-packets) is handled by RabbitMQ in a completely transparent way.
The AI-Toolkit collectes all the "nodes" that can be used for defining learning/eval modules implementing the overall AI features of any TEACHING app.

## Base Python Classes
At the moment the AI-Toolkit offers the following base classes:

- **LearningModule**: a Python class every learning module node can inherit from. It implements the basic API of a learning module node (both for inference/eval and training).
- **ESN**: general echo state network for sequential classification implemented in Tensorflow.

## Currently Available Nodes
At the moment the AI-Toolkit offers:

- **StressModule**: learning module for the stress prediction based on heart rate and electrodermal activity, implemented in Tensorflow.
  - *input topics*: *defined at the docker.compose level*
  - *output topics*: "prediction.stress.value"
  - *assumptions*: it assumes each data-packet processed has "eda" in its JSON

## How to Use Nodes
Nodes can be instantiated at the docker.compose level (i.e. defining a new yaml file with the appropriate syntax: see the Teaching app repository for examples and further instructions). When adding a new node to the app, the *input topic* (i.e. the communication channel the node will listen to) needs to be defined there. The *output topic* is hardcoded into the node logic (and listed in the section above for easy reference).

When the TEACHING app starts, the node `__init__` method will be run, followed by the `_build` method. The `__call__` implements a Python *generator*: this means that if the node is a producer, it will *yeald* data-package downstream; if it is a consuper it will loop over an *input_fn* (another generator) defined upstream in the computational graph.

## How to Implement Custom Nodes
If you want to implement a new learning/eval module you can take the *StressModule* as a reference. Overall, you just need to define the three functions mentioned before: `__init__`, `_build` (can be void) and `__call__`. The only aspect that deserves a little of attention is how to implement the call method. First, it is important to think well if the node is a producer, a consumer, or both. This should be reflected into the decorator `@TEACHINGNode(produce=True, consume=True)`. If `consume=True` then the call method should have a input parameter `input_fn` that we can loop over to get new data-packets from nodes upstream. If `produce=True` then the call method needs to *yield* a DataPacket: this will be downstreamed automatically to the (eventual) consumer nodes.

## Debug/Test Mode

*...to be added!*

## FAQ

> Can I implement more methods into my learning module?

You can create as many methods as you like: just make sure they are called withing the `__call__` method in the right order.

> How can I pre-process data before using them for training or prediction?

You can implement you own data-processing method withing the main Python class and call it before passing the data to another train/eval method. You may need to create a node entirely dedicated to pre-processing only if multiple (many) nodes in the computational graph need that preprocessed output.

