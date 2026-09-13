def is_relevent(score: float, thresold: float = 1.0) -> bool:
    return score <= thresold

if __name__ == "__main__":
    print("-----------------------------------")
    print(is_relevent(0.88))
    print(is_relevent(0.99))
    print(is_relevent(1.02))


# this will check that all 3 chunks that we are retriving is relevant or not
# retriever.py
#     → finds documents

# relevance.py
#     → decides whether retrieved results pass our relevance rule

# llm.py
#     → generates the answer
