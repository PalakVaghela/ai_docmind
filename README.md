# DOCMIND

                    USER QUESTION
                         │
                         ▼
                  ┌─────────────┐
                  │   ROUTER    │
                  └──────┬──────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        DOCUMENT QUERY         GENERAL QUERY
              │                     │
              ▼                     ▼
         RETRIEVER                OLLAMA
              │
              ▼
        RELEVANCE CHECK
          │         │
        GOOD       BAD
          │         │
          ▼         ▼
        OLLAMA    FALLBACK
          │         │
          └────┬────┘
               ▼
             END
