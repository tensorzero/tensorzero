TensorZero Demo Quick Start Guide
Setup using GitHub Codespaces

1. Get Started with Codespaces

Navigate to your GitHub repository.
Click on the "Code" button and select "Open with Codespaces".
If you don't have a Codespace, create a new one.

2. Get Your API Key

Free with Groq though fine tuning will not be accessible with this demo per Groq limitations.

Go to console.groq.com and sign up.
Create and copy your API key (starts with gsk_...).

3. Navigate to Demo Directory

Open the terminal and navigate to the demo folder:

bash

cd codespace-quickstart

4. Set Your API Key

Option A: Set in terminal (recommended)

bash
export GROQ_API_KEY="gsk-your-key-here"
Option B: Set in Docker Compose
Open docker-compose.yml and add your API key in the environment section:

yaml
GROQ_API_KEY: your_api_key_here

5. Start the Demo

In the terminal (from the github-codespace-quickstart directory), run:

bash
docker compose up -d
You will see:

text
✔ Network tensorzero-demo_default         Created                                                                                                                                                   0.0s 
✔ Container tensorzero-demo-clickhouse-1  Created                                                                                                                                                   0.0s 
✔ Container tensorzero-demo-gateway-1     Created                                                                                                                                                   0.0s 
✔ Container tensorzero-demo-ui-1          Created               

There is about a 2 minute delay before the UI window will appear - you will see an open browser button for port 4000 at lower right side of terminal or click the "ports" button and you can hover over port 4000 and select the globe icon to view the UI, it will read "open in browser" when hovered over.

6. Generate Data

Open a new terminal tab within Codespaces:

Click on the plus sign at the top right of the terminal.
Navigate to the demo directory:
bash
cd github-codespace-quickstart
Run any of these scripts:

bash

python example.py          # Basic demo
python generate_haikus.py  # Haiku generation  
python main.py             # Full demo

7. View Results

Open http://localhost:3000 to access your dashboard.
Refresh the page to see updated inferences.
That's It! 🎉
Your dashboard will show:

Inferences: API calls made
Episodes: Interaction sessions
Functions: Any function calls made
Quick Demo Loop
For generating continuous data:

bash
# Make sure you're in codespace-quickstart directory first

cd github-codespace-quickstart

for i in {1..5}; do python example.py; sleep 2; done
Note: The {"error":"Route not found: GET /"} message is expected. This is an API-focused demo—run Python scripts to generate and view results! 🐍

File Explanation
example.py: Demonstrates basic usage of the TensorZero API.
explore_api.py: Discovers available API endpoints.
generate_haikus.py: Generates haikus using TensorZero.
main.py: Comprehensive demo script showcasing core capabilities.
query_clickhouse.py: Queries ClickHouse for database insights.
tensorzero.toml: Configuration file for environment settings.
docker-compose.yml: Defines and runs multi-container Docker applications.
Key Point: All terminal commands must be run from inside the codespace-quickstart directory!


Please see tutorial video if steps are problematic for usage:
https://youtu.be/hGYBH5Fc5ZE