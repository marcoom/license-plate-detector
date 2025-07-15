# Use the official Python 3.11.13 image as the base image
FROM python:3.11.13

# Set the working directory
WORKDIR /app

# Debian/Ubuntu‑based images (python:<version> or python:<version>-slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . /app

# Install project dependencies using make
RUN make install

# Run the application
CMD ["make", "run"]
