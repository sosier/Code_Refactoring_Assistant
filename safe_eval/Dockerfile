# Use an official Python runtime as a parent image
FROM python:3.11.8-slim

# Set the working directory in the container
RUN mkdir /code
WORKDIR /code

# Copy safe_eval.py form the current directory into the container
COPY safe_eval.py .

# Run safe_eval.py when the container launches
ENTRYPOINT ["python", "safe_eval.py"]
