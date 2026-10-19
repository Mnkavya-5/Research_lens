# Failure-case checklist

How ResearchLens behaves when things go wrong, and the test that proves it.
Run all tests with `pytest -q` (no network or Ollama needed).

| Failure case | Expected behaviour | Covered by |
|---|---|---|
| **Duplicate papers** (same paper from arXiv and Crossref, different punctuation) | Merged into one, keeping the copy with a PDF | `test_dedupe.py::test_duplicates_merged_keeping_pdf`, `test_discover.py::test_results_merged_and_deduped` |
| Different papers with similar titles | Kept separate | `test_dedupe.py::test_different_papers_kept` |
| Same chunk indexed twice | Replaced, not duplicated (upsert) | `test_rag.py::test_upsert_does_not_duplicate` |
| Already-collected paper suggested by gap analysis | Not suggested again | `test_gaps.py::test_coverage_labels` |
| **Missing PDF** (Crossref, paywalled) | Skipped for download, still listed as metadata-only | `test_download.py::test_download_all_reports_errors_and_continues`, `test_failures.py::test_ingest_reports_errors_from_every_stage` |
| Download returns a login/HTML page | Rejected (not a `%PDF`), nothing saved | `test_download.py::test_html_page_rejected` |
| Download fails (404, timeout) | Error recorded, other downloads continue | `test_download.py::test_download_all_reports_errors_and_continues` |
| Scanned PDF (no text layer) | Clear "needs OCR" error, pipeline continues | `test_pdf.py::test_blank_pdf_flagged_as_scanned`, `test_failures.py::test_pipeline_skips_scanned_pdf_and_keeps_going` |
| Corrupt PDF | Clear extraction error | `test_pdf.py::test_corrupt_file_raises_extraction_error` |
| **Irrelevant search results** | Ranked below relevant ones | `test_ranking.py::test_relevant_paper_ranks_first_and_parts_explain_score`; measured: nDCG@10 0.67 (`eval/run_eval.py`) |
| Crossref figure/table records | Filtered out | `test_ranking.py::test_crossref_skips_figure_records` |
| Off-topic papers in gap suggestions | Filtered by topic similarity | `test_gaps.py::test_external_check_drops_off_topic_papers` |
| **Unsupported answers** (hallucination) | Warning shown | `test_rag.py::test_answer_warns_on_unsupported_sentence`; measured: 100% of unanswerable questions refused or warned |
| Answer without citations | "Unverified" warning | `test_rag.py::test_answer_warns_when_no_citations` |
| Citation to a source that does not exist | Flagged as invalid | `test_rag.py::test_check_citations_flags_invented_sources` |
| LLM invents a dataset/method in gap extraction | Dropped and counted | `test_gaps.py::test_extraction_drops_invented_items_and_keeps_pages` |
| LLM invents a quote in a "disagreement" | Not reported | `test_disagree.py::test_invented_quote_is_rejected` |
| LLM returns broken JSON | Handled, no crash | `test_ranking.py::test_parse_judgement`, `test_gaps.py::test_extraction_survives_bad_llm_reply`, `test_disagree.py::test_no_contradiction_and_bad_json` |
| One search API down | Other source's results returned + error shown | `test_discover.py::test_one_source_failing_keeps_the_other` |
| Temporary network error | Retried with backoff | `test_discover.py::test_retry_succeeds_after_one_failure` |
| Ollama down during query generation / judging | Falls back (topic only / baseline score) | `test_query_gen.py::test_fallback_when_ollama_down`, `test_ranking.py::test_judge_falls_back_to_abstract_and_survives_ollama_down` |
| Ollama down during chat | Clear error message in the app, no crash | `test_failures.py::test_app_shows_error_when_ollama_is_down` |
| Web page blocked by robots.txt | Not read; reason shown | `test_web.py::test_robots_disallow_is_respected` |
| Paywall / login / JavaScript-only page | Skipped with a reason | `test_web.py::test_paywalls_and_non_pages_are_skipped` |
| One web search source down | Other source still used | `test_web.py::test_search_falls_back_to_free_sources` |
| Web page already in the collection | Not suggested again | `test_agent.py::test_web_suggestions_skip_pages_already_indexed` |
| Question asked with nothing indexed | Refusal, LLM not called | `test_failures.py::test_empty_collection_refuses_without_calling_llm` |
| Adding papers after gap analysis | Ranked table still renders | `test_app.py::test_ranked_table_survives_unscored_papers_being_added` |

## Known gaps (not handled yet)
- Scanned PDFs are reported, not read (no OCR).
- arXiv versions (v1, v2) indexed at different times are stored as separate papers.
- Appendices after the references section are dropped.
- Topics share one collection, so old topics can appear in answers.
- The faithfulness check is word overlap, so it can miss paraphrased support (false alarms) and cannot detect false statements that reuse source words.
