# Database Persistence Architecture

This document describes the design behind the database layer used to store extraction artifacts. The goal is to avoid cluttering the disk with temporary markdown, JSON, or PNG files in an `artifacts/` folder, and instead use a robust database backend.

## Overview

The processing pipeline completely parses PDF documents into memory using Docling. This memory structure (`ExtractionResult`) is then persisted into a unified SQLite database (`processor.db`) using the `DatabaseProvider` interface.

To support rapid iteration and debugging, all data—including images and equations—are stored using native, easily scannable text formats.

## Schema Details

The exact SQL tables map directly to the datatypes in `src.processing.document.schema`.
- **`documents`**: Tracks the original PDF path for a given `doc_id`.
- **`chunks`**: Stores all text chunks, context breadcrumbs, and referenced metadata as JSON arrays (`headings_json`, `captions_json`).
- **`equations`**: Stores the LaTeX form or string form of equations mapped to `id`.
- **`tables`**: Contains raw `html_content` of complex tables extracted via Docling.
- **`images`**: Uses `mime_type` and `base64_data` to store the PNG bytestrings from PIL images instead of utilizing binary schemas.
- **`references_map`**: Tracks where in the markdown string specific items (images, equations, tables) exist.

### Base64 Images and Markdown In-Memory
Docling utilizes `ImageRefMode.REFERENCED` when generating Markdown, normally pointing to physical local paths. Our pipeline captures the Markdown and Image objects before they are dumped to disk. It extracts the `docling.document.pictures` natively, converts them to `.PNG` bytes, encodes them in base64, and inserts them into SQLite. The LLM can then query `processor.db` to reconstruct the multimodal assets directly.

## Inspecting the Database Manually

Because the backend relies heavily on JSON and string schemas instead of binary objects, users can quickly inspect the data pipeline outputs.

**Recommendation:** Use [DB Browser for SQLite](https://sqlitebrowser.org/) or a similar graphical interface.

1. Open `processor.db` from the project root.
2. Select **Browse Data**.
3. Select the `chunks` table to read precisely what semantic blocks the LLM will see.
4. Select the `images` table to see the base64 mapping, page location, and captions.

## Utilizing the `--use-db` Flag

To skip the expensive processing times for debugging graphs/LLM calls, you can point the app to load entirely from the DB.
`python main.py --pdf my_doc.pdf --use-db`

**Note:** If `--use-db` is set but the `doc_id` inside the `processor.db` does not match the inferred `doc_id` of the `--pdf` passed (or default document), the script will explicitly crash and exit. This provides strong guarantees against downstream poisoning using the wrong pipeline DB cache.
