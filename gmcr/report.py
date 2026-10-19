"""Executive PDF report of the current analysis."""
from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from gmcr.config import PLATFORMS, label


def _table(rows):
    s = getSampleStyleSheet()["BodyText"]
    t = Table([[Paragraph(escape(str(a)), s), Paragraph(escape(str(b)), s)] for a, b in rows],
              hAlign="LEFT", colWidths=[60 * mm, 110 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def pdf(platform: str, workload: dict, forecast: dict, impact: dict, sla_min: float, cost_per_hour: float,
        incident: str, overhead_min: float, flags: list, what_ifs: list, cv_scores: dict,
        incident_analysis: dict | None = None) -> bytes:
    """forecast: predict.predict(...) output. flags: rows of restore.factors with flag set, as dicts with
    'input', 'value', 'multiplier'. what_ifs: restore.what_if(...).to_dict('records').
    incident_analysis: optional pipeline.analyze(...) output for a live incident."""
    s = getSampleStyleSheet()
    p50 = forecast["p50_min"] + overhead_min
    p90 = forecast["p90_min"] + overhead_min
    status = "SLA BREACH" if impact["breach"] else "SLA met"
    story = [
        Paragraph("GMCR Recovery Forecast", s["Title"]),
        Paragraph(f"{PLATFORMS[platform]['name']} · {forecast['algorithm']} · {date.today().isoformat()}", s["Normal"]),
        Spacer(1, 6 * mm),
        Paragraph("Executive summary", s["Heading2"]),
        Paragraph(f"<b>{status}.</b> Worst realistic recovery (P90) is <b>{p90:.1f} min</b> against a {sla_min:.0f} min "
                  f"target. Worst-case (P90) downtime cost is <b>${impact['downtime_cost']:,.0f}</b>"
                  + (f", of which ${impact['breach_cost']:,.0f} falls beyond the SLA." if impact["breach"] else "."), s["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("Forecast", s["Heading2"]),
        _table([["Physics baseline", f"{forecast['physics_min']:.1f} min"],
                ["Model forecast P50 / P90", f"{forecast['p50_min']:.1f} / {forecast['p90_min']:.1f} min"],
                [f"Incident overhead ({incident}, assumed)", f"{overhead_min:.1f} min"],
                ["Total recovery P50 / P90", f"{p50:.1f} / {p90:.1f} min"],
                ["Cloud friction", f"{forecast['friction_min']:.1f} min"],
                ["Friction cost (part of downtime cost)", f"${impact['friction_cost']:,.0f}"],
                ["Model accuracy (5-fold CV)", f"MAE {cv_scores['mae_min']:.1f} min, R² {cv_scores['r2']:.3f}"]]),
        Spacer(1, 4 * mm),
        Paragraph("Workload", s["Heading2"]),
        _table([[label(k), v] for k, v in workload.items()] + [["Cost of downtime", f"${cost_per_hour:,.0f} per hour"]]),
        Spacer(1, 4 * mm),
        Paragraph("Why it is slow", s["Heading2"]),
    ]
    story.append(_table([[label(f["input"]), f"{f['value']} makes this restore ×{f['multiplier']:.2f} slower than typical"]
                         for f in flags]) if flags else Paragraph("No changeable input is flagged.", s["Normal"]))
    story += [Spacer(1, 4 * mm), Paragraph("Recommended changes (model what-if)", s["Heading2"])]
    story.append(_table([[label(w["input"]), f"{w['current']} → {w['change_to']}: saves {w['minutes_saved']:.0f} min "
                                             f"({w['p50_now']:.0f} → {w['p50_after']:.0f} min P50)"] for w in what_ifs])
                 if what_ifs else Paragraph("No single change saves more than 5%.", s["Normal"]))
    if incident_analysis:
        a = incident_analysis
        rows = [["Failure risk", a["risk"]["message"]]]
        if a["diagnosis"]:
            rows.append(["Likely cause", a["diagnosis"]["message"]])
        rows.append([f"Recovery plan ({a['plan']['severity']})", a["plan"]["message"]])
        story += [Spacer(1, 4 * mm), Paragraph("Live incident", s["Heading2"]), _table(rows)]
    story += [Spacer(1, 6 * mm), Paragraph(
        "Forecasts come from models trained on historical restore records and carry the error shown above. "
        "Incident overhead and playbook times are assumptions, not learned from data. Validate with a recovery "
        "drill before relying on these figures.", s["Italic"])]
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title="GMCR Recovery Forecast").build(story)
    return buf.getvalue()
