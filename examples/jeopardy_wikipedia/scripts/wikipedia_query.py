import wikipedia

def search_wikipedia(topic):
    try:
        return wikipedia.summary(topic, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e}"
    except wikipedia.exceptions.PageError:
        return "Topic not found."

if __name__ == "__main__":
    topic = "TensorFlow"
    print(search_wikipedia(topic))
