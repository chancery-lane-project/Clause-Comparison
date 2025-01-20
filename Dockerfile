# Use an official Python runtime as a base image
FROM python:3.10.15

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock to the working directory
COPY pyproject.toml poetry.lock README.md /app/

# Install Poetry (for dependency management)
RUN pip install poetry

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-root

# Copy the application code
COPY tclp/ /app/tclp/

# Set environment variables for FastAPI
ENV HOST=0.0.0.0

# Expose default ports for FastAPI
EXPOSE 8000
EXPOSE 8080

# The CMD will be overridden in docker-compose.yml
CMD ["echo", "Base image built!"]
