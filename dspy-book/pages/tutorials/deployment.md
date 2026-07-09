---
type: Web Page
title: Deployment - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/deployment
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Tutorial: Deploying your DSPy program

This guide demonstrates two potential ways to deploy your DSPy program in production: FastAPI for lightweight deployments and MLflow for more production-grade deployments with program versioning and management.

Below, we’ll assume you have the following simple DSPy program that you want to deploy. You can replace this with something more sophisticated.

## Deploying with FastAPI

FastAPI offers a straightforward way to serve your DSPy program as a REST API. This is ideal when you have direct access to your program code and need a lightweight deployment solution.

Let’s create a FastAPI application to serve your `dspy_program` defined above.

In the code above, we call `dspy.asyncify` to convert the dspy program to run in async mode for high-throughput FastAPI
deployments. Currently, this runs the dspy program in a separate thread and awaits its result.

By default, the limit of spawned threads is 8. Think of this like a worker pool.
If you have 8 in-flight programs and call it once more, the 9th call will wait until one of the 8 returns.
You can configure the async capacity using the new `async_max_workers` setting.

## Streaming, in DSPy 2.6.0+

Streaming is also supported in DSPy 2.6.0+, which can be installed via `pip install -U dspy`.

We can use `dspy.streamify` to convert the dspy program to a streaming mode. This is useful when you want to stream
the intermediate outputs (i.e. O1-style reasoning) to the client before the final prediction is ready. This uses
asyncify under the hood and inherits the execution semantics.

Write your code to a file, e.g., `fastapi_dspy.py`. Then you can serve the app with:

It will start a local server at `http://127.0.0.1:8000/`. You can test it with the python code below:

You should see the response like below:

## Deploying with MLflow

We recommend deploying with MLflow if you are looking to package your DSPy program and deploy in an isolated environment. MLflow is a popular platform for managing machine learning workflows, including versioning, tracking, and deployment.

Let’s spin up the MLflow tracking server, where we will store our DSPy program. The command below will start a local server at
`http://127.0.0.1:5000/`.

Then we can define the DSPy program and log it to the MLflow server. “log” is an overloaded term in MLflow, basically it means
we store the program information along with environment requirements in the MLflow server. This is done via the `mlflow.dspy.log_model()`
function, please see the code below:

Note

As of MLflow 2.22.0, there is a caveat that you must wrap your DSPy program in a custom DSPy Module class when deploying with MLflow.
This is because MLflow requires positional arguments while DSPy pre-built modules disallow positional arguments, e.g., `dspy.Predict`
or `dspy.ChainOfThought`. To work around this, create a wrapper class that inherits from `dspy.Module` and implement your program’s
logic in the `forward()` method, as shown in the example below.

We recommend you to set `task="llm/v1/chat"` so that the deployed program automatically takes input and generate output in
the same format as the OpenAI chat API, which is a common interface for LM applications. Write the code above into
a file, e.g. `mlflow_dspy.py`, and run it.

After you logged the program, you can view the saved information in MLflow UI. Open `http://127.0.0.1:5000/` and select
the `deploy_dspy_program` experiment, then select the run your just created, under the `Artifacts` tab, you should see the
logged program information, similar to the following screenshot:

Grab your run id from UI (or the console print when you execute `mlflow_dspy.py`), now you can deploy the logged program
with the following command:

After the program is deployed, you can test it with the following command:

You should see the response like below:

For complete guide on how to deploy a DSPy program with MLflow, and how to customize the deployment, please refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

### Best Practices for MLflow Deployment

- **Environment Management**: Always specify your Python dependencies in a- `conda.yaml`or- `requirements.txt`file.
- **Versioning**: Use meaningful tags and descriptions for your model versions.
- **Input Validation**: Define clear input schemas and examples.
- **Monitoring**: Set up proper logging and monitoring for production deployments.

For production deployments, consider using MLflow with containerization:

For a complete guide on production deployment options and best practices, refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

# Citations

1. Source page: https://dspy.ai/tutorials/deployment
