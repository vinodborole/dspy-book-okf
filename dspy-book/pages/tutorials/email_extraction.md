---
type: Web Page
title: Email Information Extraction - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/email_extraction
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Extracting Information from Emails with DSPy

This tutorial demonstrates how to build an intelligent email processing system using DSPy. We’ll create a system that can automatically extract key information from various types of emails, classify their intent, and structure the data for further processing.

## What You’ll Build

By the end of this tutorial, you’ll have a DSPy-powered email processing system that can:

- **Classify email types**(order confirmation, support request, meeting invitation, etc.)
- **Extract key entities**(dates, amounts, product names, contact info)
- **Determine urgency levels**and required actions
- **Structure extracted data**into consistent formats
- **Handle multiple email formats**robustly

## Prerequisites

- Basic understanding of DSPy modules and signatures
- Python 3.9+ installed
- OpenAI API key (or access to another supported LLM)

## Installation and Setup

## Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration[MLflow](https://mlflow.org/)is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below. ![MLflow Trace](./mlflow-tracing-email-extraction.png) 1. Install MLflow 2. Start MLflow UI in a separate terminal 3. Connect the notebook to MLflow 4. Enabling tracing. To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

## Step 1: Define Our Data Structures

First, let’s define the types of information we want to extract from emails:

## Step 2: Create DSPy Signatures

Now let’s define the signatures for our email processing pipeline:

## Step 3: Build the Email Processing Module

Now let’s create our main email processing module:

## Step 4: Running the Email Processing System

Let’s create a simple function to test our email processing system:

## Expected Output

## Next Steps

- **Add more email types**and refine classification (newsletter, promotional, etc.)
- **Add integration**with email providers (Gmail API, Outlook, IMAP)
- **Experiment with different LLMs**and optimization strategies
- **Add multilingual support**for international email processing
- **Optimization**for increasing the performance of your program

# Citations

1. Source page: https://dspy.ai/tutorials/email_extraction
