"""Synthetic DevOps incident-triage domain: diagnostic facts, decision rules, sentence templates.

The rules are ordered; the first rule that fires wins. Every rule only tests facts that are
present and only fires on positive evidence, so removing facts can never make a rule fire.
That monotonicity is what lets `relevant_facts` (the facts the winning rule tested) be a
sufficient context on their own.
"""
from __future__ import annotations

import random

from context_decisions.schemas import ASK, FactValue

Facts = dict[str, FactValue]

# --- diagnostic facts ---------------------------------------------------------------------

APPLICATION_ERRORS = ["schema mismatch", "null pointer exception", "serialization failure",
                      "missing config key"]

# Values that fire no rule. Every case starts from these before its signal is added.
BENIGN: Facts = {
    "deployment_recent": False,
    "regression_severity": "none",
    "application_errors": "none",
    "database_latency": "normal",
    "cpu_usage": "normal",
    "memory_usage": "normal",
    "packet_loss": "normal",
    "customer_impact": "low",
}
DIAGNOSTIC_KEYS = [*BENIGN, "known_bad_deployment"]


# --- rules ----------------------------------------------------------------------------------

def _degraded_subsystems(f: Facts) -> list[str]:
    keys = []
    if f.get("database_latency") == "high":
        keys.append("database_latency")
    if f.get("packet_loss") == "high":
        keys.append("packet_loss")
    keys += [k for k in ("cpu_usage", "memory_usage") if f.get(k) == "saturated"]
    return keys


def _infra_count(keys: list[str]) -> int:
    """CPU and memory saturation count as one degraded subsystem."""
    return len([k for k in keys if k not in ("cpu_usage", "memory_usage")]) + \
        any(k in ("cpu_usage", "memory_usage") for k in keys)


def decide_with_reason(f: Facts) -> tuple[str, list[str]]:
    """Return (decision, facts the decision rests on). Unknown facts are simply absent."""
    degraded = _degraded_subsystems(f)
    if f.get("customer_impact") == "high" and _infra_count(degraded) >= 2:
        return "escalate", ["customer_impact", *degraded]
    if (f.get("deployment_recent") is True and f.get("known_bad_deployment") is True
            and f.get("regression_severity") == "severe"):
        return "rollback_deployment", ["deployment_recent", "known_bad_deployment",
                                       "regression_severity"]
    if f.get("deployment_recent") is True and f.get("application_errors") not in (None, "none"):
        return "inspect_application", ["deployment_recent", "application_errors"]
    if f.get("database_latency") == "high" and f.get("cpu_usage") == "normal":
        return "inspect_database", ["database_latency", "cpu_usage"]
    if f.get("packet_loss") == "high":
        return "inspect_network", ["packet_loss"]
    saturated = [k for k in ("cpu_usage", "memory_usage") if f.get(k) == "saturated"]
    if saturated:
        return "inspect_infrastructure", saturated
    return ASK, []


def decide(f: Facts) -> str:
    return decide_with_reason(f)[0]


# --- sentence templates ---------------------------------------------------------------------

def render_fact(key: str, value: FactValue, rng: random.Random) -> str:
    r = rng
    match key, value:
        case "deployment_recent", True:
            return r.choice(["a new version was deployed {n} minutes ago",
                             "the latest deployment went out {n} minutes ago"]).format(n=r.randint(5, 50))
        case "deployment_recent", False:
            return r.choice(["the last deployment was {n} days ago",
                             "nothing has been deployed in the past {n} days"]).format(n=r.randint(3, 20))
        case "known_bad_deployment", True:
            return r.choice(["canary analysis flagged the latest release as bad",
                             "the latest release is on the known-bad release list"])
        case "known_bad_deployment", False:
            return r.choice(["canary analysis for the latest release passed",
                             "the latest release passed its canary checks"])
        case "regression_severity", "none":
            return "the error rate is at its usual baseline"
        case "regression_severity", "moderate":
            return "the error rate is {n}% above baseline".format(n=r.randint(10, 35))
        case "regression_severity", "severe":
            return r.choice(["the error rate is {n}x above baseline and SLOs are breached",
                             "the error rate jumped {n}x and the SLO budget is exhausted"]).format(n=r.randint(5, 30))
        case "application_errors", "none":
            return r.choice(["application logs show no new errors",
                             "no new exceptions appear in the application logs"])
        case "application_errors", err:
            return r.choice(["{e} errors appear in the application logs",
                             "application logs are full of {e} errors"]).format(e=err)
        case "database_latency", "normal":
            return "database query latency is normal (p95 {n} ms)".format(n=r.randint(4, 30))
        case "database_latency", "high":
            return r.choice(["database query latency is high (p95 {n} ms)",
                             "slow queries are piling up in the database (p95 {n} ms)"]).format(n=r.randint(800, 3000))
        case "cpu_usage", "normal":
            return "CPU usage on the app servers is {n}%".format(n=r.randint(20, 55))
        case "cpu_usage", "saturated":
            return "CPU on the app servers is saturated at {n}%".format(n=r.randint(96, 100))
        case "memory_usage", "normal":
            return "memory usage on the app servers is {n}%".format(n=r.randint(30, 60))
        case "memory_usage", "saturated":
            return "memory on the app servers is exhausted and pods are being OOM-killed"
        case "packet_loss", "normal":
            return "packet loss between services is {n}%".format(n=round(r.uniform(0.0, 0.1), 2))
        case "packet_loss", "high":
            return "packet loss between services is high at {n}%".format(n=r.randint(5, 20))
        case "customer_impact", "low":
            return r.choice(["only a few internal users are affected",
                             "customer impact is limited to a handful of users"])
        case "customer_impact", "high":
            return r.choice(["a large share of customers is affected",
                             "the incident affects most paying customers"])
    if key in DISTRACTORS:
        return DISTRACTORS[key]
    raise ValueError(f"no template for {key}={value!r}")


# Plausible incident-room facts with no bearing on the decision. Several share vocabulary
# with diagnostic facts (CPU, database, network, deployment) to make retrieval non-trivial.
DISTRACTORS = {
    "oncall_engineer": "the on-call engineer this week is on the platform team",
    "staging_db_migration": "the staging database was migrated to a new version last week",
    "network_maintenance": "network maintenance is scheduled for next month",
    "tls_certificate": "the TLS certificate expires in 45 days",
    "log_retention": "log retention is set to 30 days",
    "dashboard_tool": "the team uses Grafana for dashboards",
    "feature_flag": "a feature flag for dark mode was toggled yesterday",
    "backup_job": "last night's database backup completed successfully",
    "marketing_site": "the marketing website was redesigned last month",
    "sprint": "the team is in the second week of the sprint",
    "cdn": "static assets are served through a CDN",
    "k8s_version": "the cluster runs Kubernetes 1.29",
    "docs_update": "the API documentation was updated this morning",
    "new_hire": "a new engineer joined the platform team",
    "cloud_costs": "cloud costs rose 4% last month",
    "load_test": "a load test is planned for next quarter",
    "dns_ttl": "DNS TTLs were lowered last quarter",
    "staging_cpu": "CPU usage on the staging cluster is elevated",
    "read_replica": "a read replica is being provisioned in another region",
    "postmortem": "the postmortem for last month's outage is still open",
    "office_vpn": "the office VPN was slow this morning",
    "deploy_freeze": "a deployment freeze starts next Friday",
    "analytics_job": "the analytics batch job logged warnings overnight",
    "status_page": "the status page vendor renewed its contract",
}

# Domain vocabulary appended to the retrieval query: what abnormal findings look like.
# It names symptoms, never decisions, so it does not tell the retriever the answer.
RETRIEVAL_QUERY_HINTS = (
    "new deployment flagged bad canary; error rate above baseline SLO breached; "
    "exceptions errors in application logs; database query latency high slow queries; "
    "CPU saturated; memory exhausted OOM-killed; packet loss high; customers affected"
)

# Requests are sampled independently of the decision, so the request alone carries no label.
REQUESTS = [
    "The API is slow.",
    "Users report that checkout is failing intermittently.",
    "The error rate on the orders service went up.",
    "Customers are complaining about timeouts.",
    "Response times on the dashboard look bad.",
    "Something is wrong with the payments API.",
    "We are getting alerts from the API gateway.",
    "The mobile app keeps showing errors.",
    "Requests to the search service are timing out.",
    "Latency on the public API has spiked.",
    "Support is getting tickets about failed logins.",
    "The service is degraded.",
    "Our health checks are flapping.",
    "Page loads have become really slow.",
]
