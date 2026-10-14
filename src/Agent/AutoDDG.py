"""Modified from AutoDDG (https://github.com/VIDA-NYU/AutoDDG) at commit 3c59e1c, Apache License 2.0
(licenses/AutoDDG-LICENSE); the changes are listed in THIRD_PARTY.md."""
SYSTEM_MESSAGE = 'You are a helpful assistant skilled in dataset semantic analysis.'

USER_PROMPT = """You are a dataset semantic analyzer. Based on the column name and sample values, classify the column \
into multiple semantic types.
Please group the semantic types under the following categories:
'temporal', 'spatial', 'entity_type', 'domain', 'usage'.
Following are the categories:
- temporal: Does this column contain temporal information? If yes, its resolution: year, quarter, month, week, day, \
time, timestamp, or other for a resolution not in this list. If no, none.
- spatial: Does this column contain spatial information? If yes, its resolution: country, state_province, county, \
city, district (a sub-city unit such as a ward, community area, police district or census tract), postal_code, \
address, coordinates, or other for a resolution not in this list. If no, none.
- entity_type: What kind of entity does the column describe? (e.g., Person, Location, Organization, Product), in at \
most {words} words.
- domain: What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic), in at most \
{words} words.
- usage: How might the data be used, one of:
  - identifier: a key of this table or of another one, including numeric codes such as postal codes and IDs.
  - category: one of a small set of values, including numeric codes and flags (e.g., EmailPromotion 0/1/2).
  - name: the name of a person, place, organization or thing.
  - measurement: a quantity where arithmetic makes sense.
  - score: a rating or a rank.
  - free_text: text written in words, such as a description or a comment.
  - other: a use not in this list. A date or time column goes here, since its temporal resolution says what it is.
Please follow these rules:
1. Where the values contradict the column name, describe the values. Where the values alone could mean several \
things, the name decides.
2. If you are unsure about a specific category, leave it null.
3. The table's other columns are context only. Classify only the column below.

Table name: {table}
Column name: {column}
Data profile: {profile}
Sample values: {values}
The table's other columns: {others}"""
