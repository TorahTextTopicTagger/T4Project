import json

def convert_topics_to_sefaria_ids(topic_list, path_to_topics_mapping_json='./good_topic_to_sefaria_id_mapping.json'):
  if type(topic_list) != list:
    raise ValueError('must pass a list of topics into this function')
  mapping = None

  with open(path_to_topics_mapping_json, 'r') as f:
    mapping = json.load(f)

  results = list()
  for topic in topic_list:
    if topic not in mapping.keys():
      # print(topic)
      results.append(None)
    else:
      results.append(mapping[topic])
  return results
