mkdir node-rag-demo
cd node-rag-demo
npm init -y

# Install latest LangChain JS + Chroma client + OpenAI
npm install \
  langchain \
  @langchain/openai \
  chromadb \
  chromadb-default-embed \
  dotenv


#  @chroma-core/chromadb \
