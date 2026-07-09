---
type: Web Page
title: Use MCP in DSPy - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/mcp
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Tutorial: Use MCP tools in DSPy

MCP, standing for Model Context Protocol, is an open protocol that standardizes how applications provide context to LLMs. Despite some development overhead, MCP offers a valuable opportunity to share tools, resources, and prompts with other developers regardless of the technical stack you are using. Likewise, you can use the tools built by other developers without rewriting code.

In this guide, we will walk you through how to use MCP tools in DSPy. For demonstration purposes,
we will build an airline service agent that can help users book flights and modify or cancel
existing bookings. This will rely on an MCP server with custom tools, but it should be easy to generalize
to [MCP servers built by the community](https://modelcontextprotocol.io/examples).

## How to run this tutorial

This tutorial cannot be run in hosted IPython notebooks like Google Colab or Databricks notebooks. To run the code, you will need to follow the guide to write code on your local device. The code is tested on macOS and should work the same way in Linux environments.

## Install Dependencies

Before starting, let’s install the required dependencies:

## MCP Server Setup

Let’s first set up the MCP server for the airline agent, which contains:

- A set of databases
- User database, storing user information.
- Flight database, storing flight information.
- Ticket database, storing customer tickets.
- A set of tools
- fetch_flight_info: get flight information for specific dates.
- fetch_itinerary: get information about booked itineraries.
- book_itinerary: book a flight on behalf of the user.
- modify_itinerary: modify an itinerary, either through flight changes or cancellation.
- get_user_info: get user information.
- file_ticket: file a backlog ticket for human assistance.

In your working directory, create a file `mcp_server.py`, and paste the following content into
it:

Before we start the server, let’s take a look at the code.

We first create a `FastMCP` instance, which is a utility that helps quickly build an MCP server:

Then we define our data structures, which in a real-world application would be the database schema, e.g.:

Following that, we initialize our database instances. In a real-world application, these would be connectors to actual databases, but for simplicity, we just use dictionaries:

The next step is to define the tools and mark them with `@mcp.tool()` so that they are discoverable by
MCP clients as MCP tools:

The last step is spinning up the server:

Now we have finished writing the server! Let’s launch it:

## Write a DSPy Program That Utilizes Tools in MCP Server

Now that the server is running, let’s build the actual airline service agent which
utilizes the MCP tools in our server to assist users. In your working directory,
create a file named `dspy_mcp_agent.py`, and follow the guide to add code to it.

### Gather Tools from MCP Servers

We first need to gather all available tools from the MCP server and make them
usable by DSPy. DSPy provides an API [ dspy.Tool](https://dspy.ai/api/primitives/Tool/)
as the standard tool interface. Let’s convert all the MCP tools to 

`dspy.Tool`.We need to create an MCP client instance to communicate with the MCP server, fetch all available
tools, and convert them to `dspy.Tool` using the static method `from_mcp_tool`:

With the code above, we have successfully collected all available MCP tools and converted them to DSPy tools.

### Build a DSPy Agent to Handle Customer Requests

Now we will use `dspy.ReAct` to build the agent for handling customer requests. `ReAct` stands
for “reasoning and acting,” which asks the LLM to decide whether to call a tool or wrap up the process.
If a tool is required, the LLM takes responsibility for deciding which tool to call and providing
the appropriate arguments.

As usual, we need to create a `dspy.Signature` to define the input and output of our agent:

And choose an LM for our agent:

Then we create the ReAct agent by passing the tools and signature into the `dspy.ReAct` API. We can now
put together the complete code script:

Note that we must call `react.acall` because MCP tools are async by default. Let’s execute the script:

You should see output similar to this:

The `trajectory` field contains the entire thinking and acting process. If you’re curious about what’s happening
under the hood, check out the [Observability Guide](https://dspy.ai/tutorials/observability/) to set up MLflow,
which visualizes every step happening inside `dspy.ReAct`!

## Conclusion

In this guide, we built an airline service agent that utilizes a custom MCP server and the `dspy.ReAct` module. In the context
of MCP support, DSPy provides a simple interface for interacting with MCP tools, giving you the flexibility to implement
any functionality you need.

# Citations

1. Source page: https://dspy.ai/tutorials/mcp
