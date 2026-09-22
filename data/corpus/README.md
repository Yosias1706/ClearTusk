# Corpus layout

| Path | Contents |
|---|---|
| `annotations.csv` | Master annotation table: `Selection, Sound_file, Start_time, End_time, Call_type` |
| `raw/` | Source field recordings, named `<session>_<interference>_<n>.wav` |
| `calls/` | One clip per annotated call (`call_<selection>.wav`) |
| `context/` | The same call with a 2 s margin either side (`call_<selection>_context.wav`) |
| `noise/` | The widest call-free gap next to each call (`call_<selection>_noise.wav`) |

`calls/`, `context/`, and `noise/` are regenerated from `raw/` and
`annotations.csv` by `cleartusk prepare-corpus`; the interference type shown in
the dashboard is parsed from the source filename.
