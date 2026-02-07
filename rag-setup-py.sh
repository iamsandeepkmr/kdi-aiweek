# Create and activate virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install latest libraries
pip install \
  langchain \
  langchain-openai \
  langchain-chroma \
  chromadb \
  tiktoken \
  python-dotenv \
  langchain-text-splitters \
  langchain-core \
  langchain_community \
  ddgs \
  langgraph
