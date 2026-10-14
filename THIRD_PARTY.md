# Third-party material

## AutoDDG

- Repository: https://github.com/VIDA-NYU/AutoDDG
- Commit: 3c59e1c466a5c0ad3e9126f8a09023815b437aef
- Licence: Apache License 2.0, full text in `licenses/AutoDDG-LICENSE`
- Used: the semantic profiler's prompt (`src/autoddg/configurations/prompts.yaml`, entry `semantic_profiler`), vendored
  and adapted in `src/Agent/AutoDDG.py`. The package itself is not installed.
- Changes from the source (Apache-2.0, section 4(b)):
  - Typed output: the template's pseudo-dict and response_example and rules 1 to 6 (the JSON format) are gone, since
    structured output enforces the schema (Pydantic AI states the schema in the instructions); the template's questions
    stay, one line per category, introduced as the categories rather than as the template.
  - Fields: the keys are temporal, spatial, entity_type, domain and usage. 'Data Format', named in the category list
    with no template entry, is dropped, and so are the isTemporal and isSpatial flags: none stands for no temporal or
    spatial content.
  - Closed lists, this design's own: the temporal and spatial resolutions and the usage values replace the source's open
    examples, each with "other" for a unit or use the list does not name; every usage value has a one-line definition,
    and a date column goes under other.
  - Entity type and domain stay free text, in at most a configured number of words.
  - Unsure: rule 7's empty string is null; every field is nullable.
  - No description field.
  - Two added instructions, this design's own: rule 1, "Where the values contradict the column name, describe the values.
    Where the values alone could mean several things, the name decides.", and rule 3, that the other columns are context
    only and only the given column is classified.
  - Input: the source's column name and five sampled rows' values (unquoted) become the table name, the column's
    data-profile line (type, counts, statistics) and its data-profile example values, quoted. The table's other columns'
    names and types are added as context only, this design's own addition: Name in vStoreWithContacts is a store's name
    only because of the columns beside it.
