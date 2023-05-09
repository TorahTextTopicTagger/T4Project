from convert_topics_to_sefaria_ids import convert_topics_to_sefaria_ids

prompt = 'Enter space-separated list of topics to convert (or "stop" to stop): '
topics = input(prompt)
while topics != 'stop':
    print(convert_topics_to_sefaria_ids(topics.split()))
    topics = input(prompt)