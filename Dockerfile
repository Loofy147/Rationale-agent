# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency files to the working directory
COPY pyproject.toml poetry.lock* ./

# Install poetry
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Copy the rest of the application's source code from your host to your image filesystem.
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "mas.main:app", "--host", "0.0.0.0", "--port", "8000"]
