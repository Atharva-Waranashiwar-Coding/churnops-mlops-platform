"""Prometheus metrics for API and inference observability."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

from churnops.inference.models import PredictionRecord

_REQUEST_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
_PREDICTION_BATCH_BUCKETS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
_PROBABILITY_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


class InferenceMetrics:
    """Own the Prometheus registry and metric definitions for one API process."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests_total = Counter(
            "churnops_api_http_requests_total",
            "Total number of HTTP requests handled by the inference API.",
            labelnames=("method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "churnops_api_http_request_duration_seconds",
            "HTTP request latency in seconds for the inference API.",
            labelnames=("method", "route"),
            buckets=_REQUEST_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.http_request_failures_total = Counter(
            "churnops_api_http_request_failures_total",
            "Total number of HTTP requests that completed with 4xx or 5xx responses.",
            labelnames=("method", "route", "status_family"),
            registry=self.registry,
        )
        self.inference_model_load_total = Counter(
            "churnops_inference_model_load_total",
            "Total number of model load attempts performed by the inference service.",
            labelnames=("model_source", "result"),
            registry=self.registry,
        )
        self.prediction_requests_total = Counter(
            "churnops_inference_prediction_requests_total",
            "Total number of successful batch prediction requests served.",
            labelnames=("model_source",),
            registry=self.registry,
        )
        self.prediction_records_total = Counter(
            "churnops_inference_prediction_records_total",
            "Total number of prediction records emitted by the inference service.",
            labelnames=("model_source", "predicted_class", "predicted_churn"),
            registry=self.registry,
        )
        self.prediction_batch_size = Histogram(
            "churnops_inference_prediction_batch_size",
            "Batch sizes observed by the inference service.",
            labelnames=("model_source",),
            buckets=_PREDICTION_BATCH_BUCKETS,
            registry=self.registry,
        )
        self.churn_probability = Histogram(
            "churnops_inference_churn_probability",
            "Distribution of positive-class churn probabilities returned by the model.",
            labelnames=("model_source",),
            buckets=_PROBABILITY_BUCKETS,
            registry=self.registry,
        )

    def record_http_request(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record the outcome and latency of a single HTTP request."""

        normalized_method = method.upper()
        normalized_route = route or "/unmatched"
        normalized_status_code = str(status_code)

        self.http_requests_total.labels(
            method=normalized_method,
            route=normalized_route,
            status_code=normalized_status_code,
        ).inc()
        self.http_request_duration_seconds.labels(
            method=normalized_method,
            route=normalized_route,
        ).observe(duration_seconds)

        if status_code >= 400:
            self.http_request_failures_total.labels(
                method=normalized_method,
                route=normalized_route,
                status_family=_status_family(status_code),
            ).inc()

    def record_model_load(self, model_source: str, result: str) -> None:
        """Record a model load attempt outcome."""

        self.inference_model_load_total.labels(
            model_source=model_source,
            result=result,
        ).inc()

    def record_prediction_batch(
        self,
        model_source: str,
        predictions: list[PredictionRecord],
    ) -> None:
        """Record prediction request, batch size, and output distribution metrics."""

        self.prediction_requests_total.labels(model_source=model_source).inc()
        self.prediction_batch_size.labels(model_source=model_source).observe(len(predictions))

        for prediction in predictions:
            self.prediction_records_total.labels(
                model_source=model_source,
                predicted_class=str(prediction.predicted_class),
                predicted_churn=str(prediction.predicted_churn).lower(),
            ).inc()
            if prediction.churn_probability is not None:
                self.churn_probability.labels(model_source=model_source).observe(
                    prediction.churn_probability
                )


def _status_family(status_code: int) -> str:
    """Collapse HTTP statuses into low-cardinality families for failure metrics."""

    if 400 <= status_code < 500:
        return "client_error"
    if status_code >= 500:
        return "server_error"
    return "success"
