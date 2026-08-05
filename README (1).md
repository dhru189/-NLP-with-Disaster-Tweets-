# Disaster Tweet Classifier

> NLP-powered web dashboard for classifying real disaster tweets from figurative language using the Kaggle NLP with Disaster Tweets Dataset.

**Domain:** Social Media / Civic  
**Dataset:** [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)  
**Tech Stack:** Next.js 16, TypeScript, Prisma, Recharts, Framer Motion, shadcn/ui, z-ai-web-dev-sdk

---

## Problem Statement

During disasters, emergency services are overwhelmed with social media data. It is critical to quickly separate tweets reporting real emergencies from those using disaster-related words figuratively (e.g., "My exam results are a complete disaster lol"). This project builds an NLP classifier that identifies real disaster tweets to help emergency teams respond faster.

### Input / Output

| Input | Output |
|-------|--------|
| "Massive flood in Ahmedabad — roads submerged near Sardar Bridge." | **REAL DISASTER** |
| "My exam results are a complete disaster lol" | **NOT A DISASTER** |
| "Forest fire near La Ronge Sask. Canada" | **REAL DISASTER** |
| "The new Avengers movie is going to be bomb!" | **NOT A DISASTER** |

---

## Approach

The project follows a standard NLP text classification pipeline:

1. **Clean** — Remove URLs, emojis, HTML tags, and punctuation using regex
2. **Tokenize** — Split text into words, remove stopwords, lemmatize
3. **TF-IDF Vectorization** — Extract feature vectors from cleaned text
4. **Classify** — AI-powered classification (LLM via z-ai-web-dev-sdk)
5. **Evaluate** — Confidence scores and explanations for each prediction

---

## Features

### 1. Overview Dashboard
- Problem statement and I/O examples
- Key dataset statistics (total tweets, class distribution, average length)
- Visual NLP pipeline stepper

### 2. Exploratory Data Analysis (EDA)
- **Target Distribution** — Donut pie chart showing class balance (Real Disaster vs Not a Disaster)
- **Tweet Length Distribution** — Bar chart comparing character length ranges by class
- **Word Count Distribution** — Bar chart comparing word count ranges by class
- **Top Words** — Horizontal bar charts of the 15 most frequent words per class (stopwords removed)
- **Common Bigrams** — Horizontal bar charts of the 15 most frequent bigrams per class

### 3. Text Preprocessing Pipeline
- Interactive 8-step visual pipeline:
  - Original Text
  - Remove URLs (`https?://\S+`)
  - Remove Emojis (Unicode regex)
  - Remove HTML Tags
  - Remove Punctuation
  - Lowercase Conversion
  - Remove Stopwords (NLTK-style list)
  - Tokenized Result (displayed as word badges)
- Step-by-step animated reveal with Framer Motion

### 4. Live Tweet Classifier
- Real-time AI-powered classification using LLM (z-ai-web-dev-sdk)
- Confidence score with animated progress bar
- Natural language explanation for each prediction
- 4 clickable sample tweets for quick testing
- Prediction history with timestamps

### 5. Dataset Browser
- Search tweets by text content
- Filter by class (All / Real Disaster / Not a Disaster)
- Paginated table with ID, Keyword, Location, Text, and Label columns
- Click-to-expand for full tweet text

---

## Project Structure

```
my-project/
├── prisma/
│   ├── schema.prisma              # Database schema (Tweet, Prediction models)
│   └── (migrations)
├── seed.ts                        # Database seeder with Kaggle sample data
├── src/
│   ├── app/
│   │   ├── layout.tsx              # Root layout with QueryProvider
│   │   ├── page.tsx                # Single-page app with tab routing
│   │   ├── globals.css             # Global styles + disaster theme
│   │   └── api/
│   │       ├── tweets/
│   │       │   ├── route.ts              # GET - Paginated tweet listing
│   │       │   └── eda/
│   │       │       ├── distribution/route.ts  # GET - Target & length distributions
│   │       │       ├── word-freq/route.ts     # GET - Top words per class
│   │       │       └── ngrams/route.ts        # GET - Top bigrams per class
│   │       │   └── stats/route.ts            # GET - Dataset statistics
│   │       ├── classify/route.ts      # POST - LLM tweet classification
│   │       ├── preprocess/route.ts    # POST - Text preprocessing pipeline
│   │       └── predictions/route.ts   # GET - Recent classification history
│   ├── components/
│   │   ├── disaster/
│   │   │   ├── Header.tsx             # Sticky header + tab navigation
│   │   │   ├── Footer.tsx             # Sticky footer
│   │   │   ├── OverviewTab.tsx        # Dashboard overview tab
│   │   │   ├── EDATab.tsx             # Exploratory data analysis tab
│   │   │   ├── PreprocessingTab.tsx   # Text preprocessing demo tab
│   │   │   ├── ClassifierTab.tsx      # Live AI classifier tab
│   │   │   ├── DatasetBrowserTab.tsx  # Dataset browser tab
│   │   │   ├── types.ts               # TypeScript interfaces
│   │   │   └── query-provider.tsx     # TanStack Query provider
│   │   └── ui/                        # shadcn/ui components
│   └── lib/
│       ├── db.ts                     # Prisma client singleton
│       └── utils.ts                  # Utility functions
├── db/
│   └── custom.db                    # SQLite database
└── package.json
```

---

## Database Schema

### Tweet Model
| Field | Type | Description |
|-------|------|-------------|
| `id` | String (cuid) | Primary key |
| `tweetId` | Int | Original Kaggle tweet ID |
| `keyword` | String? | Disaster keyword tag |
| `location` | String? | Tweet location |
| `text` | String | Full tweet text |
| `target` | Int | 1 = Real Disaster, 0 = Not a Disaster |
| `createdAt` | DateTime | Timestamp |

### Prediction Model
| Field | Type | Description |
|-------|------|-------------|
| `id` | String (cuid) | Primary key |
| `tweetText` | String | Input tweet text |
| `label` | String | "Real Disaster" or "Not a Disaster" |
| `confidence` | Float | Classification confidence (0.0–1.0) |
| `explanation` | String | AI reasoning explanation |
| `createdAt` | DateTime | Timestamp |

---

## API Endpoints

### GET `/api/tweets/stats`
Returns dataset statistics: total count, class distribution percentages, average tweet length, average word count, top 15 keywords, and top 10 locations.

### GET `/api/tweets/eda/distribution`
Returns target distribution labels/values, tweet length distribution by class (ranges: 0–20, 21–40, 41–60, 61–80, 81–100, 100+), and word count distribution by class (ranges: 1–5, 6–10, 11–15, 16–20, 20+).

### GET `/api/tweets/eda/word-freq`
Returns top 30 most frequent words for disaster and non-disaster tweets after stopword removal and text cleaning.

### GET `/api/tweets/eda/ngrams`
Returns top 20 most frequent bigrams for disaster and non-disaster tweets.

### GET `/api/tweets?page=&limit=&search=&target=`
Returns paginated tweet listing with search and filter support. Supports filtering by text content and target class.

### POST `/api/classify`
Classifies a tweet using the LLM. Request body: `{ "text": "tweet text" }`. Returns label, confidence score, and explanation. Saves the prediction to the database.

### POST `/api/preprocess`
Runs the 8-step text preprocessing pipeline. Request body: `{ "text": "tweet text" }`. Returns each transformation step result.

### GET `/api/predictions`
Returns the 20 most recent classification predictions, ordered by creation date descending.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | Next.js 16 (App Router) |
| **Language** | TypeScript 5 |
| **Styling** | Tailwind CSS 4 |
| **UI Components** | shadcn/ui (New York style) |
| **Database** | Prisma ORM + SQLite |
| **Data Fetching** | TanStack Query v5 |
| **Charts** | Recharts 2 |
| **Animations** | Framer Motion 12 |
| **Icons** | Lucide React |
| **AI Classification** | z-ai-web-dev-sdk (LLM) |
| **State Management** | Zustand 5 |

---

## Getting Started

### Prerequisites

- [Bun](https://bun.sh/) (or Node.js 18+)
- SQLite (included with Prisma)

### Installation

```bash
# 1. Install dependencies
bun install

# 2. Set up the database
bun run db:push

# 3. (Optional) Seed the database with sample tweets
bun run seed.ts

# 4. Start the development server
bun run dev
```

The application will be available at `http://localhost:3000`.

### Environment Variables

The project uses a `DATABASE_URL` environment variable for SQLite. This is typically set to:

```env
DATABASE_URL="file:./db/custom.db"
```

---

## Dataset

This project uses a representative sample from the **[Kaggle NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)** dataset:

- **120 tweets** total (54 real disasters, 66 non-disasters)
- Covers diverse disaster types: earthquakes, floods, hurricanes, wildfires, tsunamis, tornadoes, volcanic eruptions, and more
- Includes figurative usage examples: colloquial expressions, movie references, exaggerations
- Fields: `id`, `keyword`, `location`, `text`, `target`

### Original Dataset Stats (Kaggle)

| Split | Records | Features |
|-------|---------|----------|
| Train | 7,613 | id, keyword, location, text, target |
| Test | 3,263 | id, keyword, location, text |

---

## References

This project was inspired by and references the following works:

- [Kaggle NLP with Disaster Tweets Competition](https://www.kaggle.com/c/nlp-getting-started)
- [BERT Research by Chris McCormick](https://www.youtube.com/watch?v=FKlPCK1uFrc) — BERT fine-tuning tutorial
- [The Official BERT Paper](https://arxiv.org/pdf/1810.04805.pdf)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [EDA for NLP by Shahul Es](https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools)
- [NLP EDA by Kamil Mysiak](https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d)

---

## License

This project is for educational and demonstration purposes. The dataset is provided by Kaggle under its competition terms.
