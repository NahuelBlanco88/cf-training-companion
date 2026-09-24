FROM python:3.11-slim
WORKDIR /service
COPY mcp_bridge/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY mcp_bridge/ ./mcp_bridge/
EXPOSE 8080
CMD ["python", "-m", "mcp_bridge.server"]
