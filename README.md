# UdaPlay

UdaPlay is an AI research assistant for video game questions. It is designed for executives, analysts, and gamers who want to ask natural language questions such as:

- "Who developed FIFA 21?"
- "When was God of War Ragnarok released?"
- "What platform was Pokemon Red launched on?"
- "What is Rockstar Games working on right now?"

The assistant first searches its local game knowledge base. When the local dataset is not enough, it can use Tavily web search to gather current information, preserve useful findings in memory, and generate a structured answer.

## What It Does

UdaPlay combines local retrieval, web search, and answer evaluation:

- Answers questions about game titles, release years, platforms, genres, descriptions, and publishers.
- Uses retrieval augmented generation over the local `games/` dataset as the first source of truth.
- Falls back to Tavily web search when internal knowledge is missing or confidence is low.
- Evaluates whether retrieved information is useful enough to answer the user's question.
- Produces clear responses with confidence levels and source-aware context.
- Can combine internal game records and external web results when a complete answer needs both.

## Project Layout

```text
.
├── games/                         # Local JSON game records
├── lib/                           # Agent, RAG, memory, LLM, parser, and tool helpers
├── Udaplay_01_solution_project.ipynb
├── Udaplay_02_solution_project.ipynb
├── requirements.txt
├── .env.example
└── README.md
```

The notebooks are the main application workflow:

- `Udaplay_01_solution_project.ipynb` builds the local ChromaDB-backed game knowledge base.
- `Udaplay_02_solution_project.ipynb` builds the research agent that retrieves, evaluates, searches the web, and answers questions.

## Setup

Use Python 3.11 or newer. This repo was prepared with a local `.venv` virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file in the repository root. You can start from `.env.example` and replace the placeholder values:

```env
OPENAI_API_KEY="your-openai-key"
TAVILY_API_KEY="your-tavily-key"
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```

The real `.env` file is ignored by Git so API keys stay local.

## Using UdaPlay

Open the notebooks in order and run them from the repository root:

1. Run `Udaplay_01_solution_project.ipynb` to load the game JSON files into a local vector database.
2. Run `Udaplay_02_solution_project.ipynb` to configure the research agent and its tools.
3. Ask game-industry questions in natural language.

Good starter questions include:

- "When was Pokemon Gold and Silver released?"
- "Which one was the first 3D platformer Mario game?"
- "Was Mortal Kombat X released for PlayStation 5?"
- "What is Rockstar Games working on right now?"

## How Answers Are Produced

UdaPlay follows a two-tier retrieval flow:

1. Search the local game dataset with RAG.
2. Evaluate whether the retrieved records are sufficient.
3. Search the web with Tavily when local confidence is low.
4. Parse and preserve useful information in long-term memory.
5. Return a readable answer with relevant source context and confidence.

## License

See [LICENSE.md](LICENSE.md).
