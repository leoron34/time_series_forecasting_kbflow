FROM python:3.12
WORKDIR /app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .
EXPOSE 5000

# Setup an app user so the container doesn't run as the root user

CMD ["./run.sh"]