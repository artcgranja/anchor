"""Tests for OTLP exporters."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from astro_context.observability.models import Span, SpanKind, TraceRecord
from astro_context.observability.otlp import (
    OTLPMetricsExporter,
    OTLPSpanExporter,
    _convert_record_to_spans,
    _convert_span,
    _datetime_to_ns,
    _map_span_kind,
)


def _hide_otel(modules: dict[str, object]) -> dict[str, object]:
    """Return *modules* with every ``opentelemetry`` key mapped to ``None``.

    Passing this to ``patch.dict(sys.modules, ...)`` makes any
    ``import opentelemetry.*`` inside the patched block raise
    ``ImportError``, regardless of whether the package is really installed.
    """
    blocked = {k: None for k in list(modules) if k.startswith("opentelemetry")}
    return blocked


class TestOTLPSpanExporter:
    """Tests for the OTLPSpanExporter class."""

    def test_import_error_without_otel(self) -> None:
        """Verify ImportError with clear message when OTel not installed."""
        with patch.dict(sys.modules, _hide_otel(sys.modules)):
            with pytest.raises(ImportError, match="opentelemetry"):
                OTLPSpanExporter()

    def test_import_error_custom_endpoint(self) -> None:
        """Verify ImportError even with custom parameters."""
        with patch.dict(sys.modules, _hide_otel(sys.modules)):
            with pytest.raises(ImportError, match="pip install astro-context"):
                OTLPSpanExporter(
                    endpoint="http://collector:4318",
                    service_name="my-service",
                    headers={"Authorization": "Bearer token"},
                )


class TestOTLPMetricsExporter:
    """Tests for the OTLPMetricsExporter class."""

    def test_import_error_without_otel(self) -> None:
        """Verify ImportError with clear message when OTel not installed."""
        with patch.dict(sys.modules, _hide_otel(sys.modules)):
            with pytest.raises(ImportError, match="opentelemetry"):
                OTLPMetricsExporter()

    def test_import_error_custom_endpoint(self) -> None:
        """Verify ImportError even with custom parameters."""
        with patch.dict(sys.modules, _hide_otel(sys.modules)):
            with pytest.raises(ImportError, match="pip install astro-context"):
                OTLPMetricsExporter(
                    endpoint="http://collector:4318",
                    service_name="my-service",
                )


class TestConvertSpan:
    """Tests for the _convert_span static helper."""

    def test_convert_single_span(self) -> None:
        """A single span converts to a dict with all expected keys."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="retrieve",
            kind=SpanKind.RETRIEVAL,
            start_time=now,
            end_time=datetime(2025, 1, 15, 12, 0, 1, tzinfo=UTC),
            duration_ms=1000.0,
            status="ok",
            attributes={"top_k": 10},
        )
        result = _convert_span(span)

        assert result["name"] == "retrieve"
        assert result["span_id"] == "span-1"
        assert result["trace_id"] == "trace-1"
        assert result["parent_span_id"] is None
        assert result["kind"] == "CLIENT"
        assert result["status"] == "ok"
        assert result["attributes"] == {"top_k": 10}
        assert result["duration_ms"] == 1000.0
        assert "start_time_ns" in result
        assert "end_time_ns" in result

    def test_span_without_end_time(self) -> None:
        """A span without end_time omits end_time_ns from the dict."""
        span = Span(
            span_id="span-2",
            trace_id="trace-1",
            name="format",
            kind=SpanKind.FORMATTING,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        result = _convert_span(span)

        assert "end_time_ns" not in result
        assert "duration_ms" not in result

    def test_span_with_parent(self) -> None:
        """A span with a parent_span_id preserves it."""
        span = Span(
            span_id="span-child",
            parent_span_id="span-parent",
            trace_id="trace-1",
            name="child",
            kind=SpanKind.MEMORY,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        result = _convert_span(span)
        assert result["parent_span_id"] == "span-parent"

    def test_span_with_events(self) -> None:
        """A span with events includes them in the dict."""
        span = Span(
            span_id="span-3",
            trace_id="trace-1",
            name="ingest",
            kind=SpanKind.INGESTION,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            events=[{"name": "chunk_created", "timestamp": "2025-01-15T12:00:00Z"}],
        )
        result = _convert_span(span)
        assert result["events"] == [
            {"name": "chunk_created", "timestamp": "2025-01-15T12:00:00Z"}
        ]

    def test_span_without_events(self) -> None:
        """A span without events omits the events key."""
        span = Span(
            span_id="span-4",
            trace_id="trace-1",
            name="query",
            kind=SpanKind.QUERY_TRANSFORM,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        result = _convert_span(span)
        assert "events" not in result

    def test_span_attributes_preserved(self) -> None:
        """Attributes are carried over from the span to the dict."""
        attrs = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        span = Span(
            span_id="span-5",
            trace_id="trace-1",
            name="llm_call",
            kind=SpanKind.PIPELINE,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            attributes=attrs,
        )
        result = _convert_span(span)
        assert result["attributes"] == attrs
        # Verify it's a copy, not the same dict
        assert result["attributes"] is not span.attributes


class TestSpanKindMapping:
    """Tests for _map_span_kind."""

    def test_retrieval_maps_to_client(self) -> None:
        assert _map_span_kind(SpanKind.RETRIEVAL) == "CLIENT"

    def test_pipeline_maps_to_internal(self) -> None:
        assert _map_span_kind(SpanKind.PIPELINE) == "INTERNAL"

    def test_ingestion_maps_to_producer(self) -> None:
        assert _map_span_kind(SpanKind.INGESTION) == "PRODUCER"

    def test_reranking_maps_to_internal(self) -> None:
        assert _map_span_kind(SpanKind.RERANKING) == "INTERNAL"

    def test_formatting_maps_to_internal(self) -> None:
        assert _map_span_kind(SpanKind.FORMATTING) == "INTERNAL"

    def test_memory_maps_to_internal(self) -> None:
        assert _map_span_kind(SpanKind.MEMORY) == "INTERNAL"

    def test_query_transform_maps_to_internal(self) -> None:
        assert _map_span_kind(SpanKind.QUERY_TRANSFORM) == "INTERNAL"


class TestDatetimeToNs:
    """Tests for _datetime_to_ns."""

    def test_utc_datetime(self) -> None:
        """UTC datetime converts to correct nanoseconds."""
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        ns = _datetime_to_ns(dt)
        assert ns == int(dt.timestamp() * 1_000_000_000)

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetime is treated as UTC."""
        dt_naive = datetime(2025, 1, 15, 12, 0, 0)
        dt_utc = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        assert _datetime_to_ns(dt_naive) == _datetime_to_ns(dt_utc)

    def test_epoch(self) -> None:
        """Unix epoch converts to zero nanoseconds."""
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert _datetime_to_ns(dt) == 0


class TestConvertRecordToSpans:
    """Tests for _convert_record_to_spans."""

    def test_empty_record(self) -> None:
        """An empty trace record yields an empty list."""
        record = TraceRecord(trace_id="trace-empty")
        result = _convert_record_to_spans(record)
        assert result == []

    def test_single_span_record(self) -> None:
        """A record with one span yields a single-element list."""
        span = Span(
            span_id="s1",
            trace_id="t1",
            name="retrieve",
            kind=SpanKind.RETRIEVAL,
            start_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        record = TraceRecord(trace_id="t1", spans=[span])
        result = _convert_record_to_spans(record)

        assert len(result) == 1
        assert result[0]["name"] == "retrieve"
        assert result[0]["trace_id"] == "t1"

    def test_multiple_spans_record(self) -> None:
        """A record with multiple spans converts all of them."""
        spans = [
            Span(
                span_id=f"s{i}",
                trace_id="t1",
                name=f"op-{i}",
                kind=SpanKind.PIPELINE,
                start_time=datetime(2025, 1, 15, 12, 0, i, tzinfo=UTC),
            )
            for i in range(5)
        ]
        record = TraceRecord(trace_id="t1", spans=spans)
        result = _convert_record_to_spans(record)

        assert len(result) == 5
        assert [r["name"] for r in result] == [f"op-{i}" for i in range(5)]

    def test_timestamps_converted(self) -> None:
        """Datetime objects are serialized to nanoseconds."""
        start = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        end = datetime(2025, 1, 15, 12, 0, 5, tzinfo=UTC)
        span = Span(
            span_id="s1",
            trace_id="t1",
            name="op",
            kind=SpanKind.PIPELINE,
            start_time=start,
            end_time=end,
            duration_ms=5000.0,
        )
        record = TraceRecord(trace_id="t1", spans=[span])
        result = _convert_record_to_spans(record)

        assert result[0]["start_time_ns"] == _datetime_to_ns(start)
        assert result[0]["end_time_ns"] == _datetime_to_ns(end)
