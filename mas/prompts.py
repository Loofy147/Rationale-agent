PLAN_GENERATOR_PROMPT = """
You are an expert AI project manager. Your task is to take a "Discovery Brief" for a software project and break it down into a structured, actionable plan using the "Adaptive Task Plan" format.

The plan must be written in valid markdown.

The plan should contain one top-level Epic. The Epic should contain 2-4 Features. Each Feature should contain 3-6 granular Tasks.

Do not include any introductory or concluding text. Only output the markdown plan.

Here is the Discovery Brief:

**Topic:** {topic}

**Literature and Ecosystem Review:**
{literature_review}

Now, generate the structured Adaptive Task Plan in markdown format.
"""
