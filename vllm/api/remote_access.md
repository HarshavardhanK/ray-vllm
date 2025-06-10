# Accessing Remote vLLM Server from Local Machine

This guide explains how to access a vLLM server running on a remote protected cluster (mlnlp) from your local machine.

## Prerequisites
- SSH access to the remote server (mlnlp)
- vLLM server running on the remote machine
- Local machine with Python and required packages installed

## Step 1: Start the vLLM Server on Remote Machine
On the remote server (mlnlp), start the vLLM server:
```bash
conda activate vllm-env
python vllm/api/server.py
```
The server will start on port 8000 by default.

## Step 2: Create SSH Tunnel
On your local machine (not on mlnlp), open a terminal and run:
```bash
ssh -L 8000:localhost:8000 mlnlp
```
This command:
- Creates a secure tunnel from your local port 8000 to port 8000 on the remote server
- Keeps the connection open (keep this terminal window open while using the server)
- Forwards all local traffic on port 8000 to the remote server

## Step 3: Use the Server Locally
Now you can use the vLLM server as if it were running locally. Here's an example using the OpenAI client:

```python
from openai import OpenAI

#Configure client to use local port (which is forwarded to remote)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  #or your API key if you've set one
)

#Test the connection
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## Important Notes
1. Keep the SSH tunnel terminal window open while using the server
2. All requests to `localhost:8000` on your local machine will be forwarded to the remote server
3. The actual processing happens on the remote server, but you interact with it as if it were local
4. If you need to use a different port, modify both the server port and the SSH tunnel port accordingly

## Troubleshooting
- If you can't connect, ensure:
  - The vLLM server is running on the remote machine
  - The SSH tunnel is active
  - You're using the correct port numbers
  - Your local machine can reach the remote server via SSH 