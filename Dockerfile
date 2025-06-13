# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (the 'app' directory) into the container
COPY ./app /usr/src/app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables (can be overridden by docker-compose)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["flask", "run"]
