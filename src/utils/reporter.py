import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Parses daily JSONL audit logs and generates comprehensive Markdown summary reports
    detailing unique drone targets, detection counts, coordinate trajectories, and snapshot paths.
    """
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read/parse configuration: {e}")
            raise

        logging_config = self.config.get("logging", {})
        self.log_dir = Path(logging_config.get("log_dir", "logs"))

        reporting_config = self.config.get("reporting", {})
        self.enabled = reporting_config.get("enabled", True)
        self.output_dir = Path(reporting_config.get("output_dir", "logs"))

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(self, date_str: str | None = None) -> Path | None:
        if not self.enabled:
            logger.info("Report generation is disabled in configuration.")
            return None

        if date_str is None:
            now_utc = datetime.now(timezone.utc)
            date_str = now_utc.strftime("%Y-%m-%d")

        log_file = self.log_dir / f"drone_events_{date_str}.jsonl"
        if not log_file.exists():
            logger.warning(f"No log file found for date '{date_str}' at: {log_file.absolute()}")
            return None

        records = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read audit log file '{log_file}': {e}")
            return None

        if not records:
            logger.info(f"Log file '{log_file}' is empty. Skipping report generation.")
            return None

        # Aggregate metrics per track_id
        tracks_summary = {}
        for rec in records:
            tid = rec.get("track_id", -1)
            ts = rec.get("timestamp", "")
            conf = rec.get("confidence", 0.0)
            bbox = rec.get("bbox", [])
            center_px = rec.get("center_px", [])
            center_norm = rec.get("center_norm", [])
            snapshot_path = rec.get("snapshot_path", None)
            geo = rec.get("geo", None)

            if tid not in tracks_summary:
                tracks_summary[tid] = {
                    "first_seen": ts,
                    "last_seen": ts,
                    "count": 0,
                    "max_conf": conf,
                    "trajectory_px": [],
                    "trajectory_geo": [],
                    "snapshots": []
                }

            t = tracks_summary[tid]
            t["last_seen"] = ts
            t["count"] += 1
            if conf > t["max_conf"]:
                t["max_conf"] = conf

            if center_px:
                t["trajectory_px"].append(center_px)

            if geo and geo.get("lat") is not None and geo.get("lon") is not None:
                t["trajectory_geo"].append({**geo, "timestamp": ts})

            if snapshot_path and snapshot_path not in t["snapshots"]:
                t["snapshots"].append(snapshot_path)

        # Build Markdown document
        report_path = self.output_dir / f"drone_report_{date_str}.md"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# 🛸 Drone Detection & Tracking Summary Report",
            f"",
            f"**Report Date:** {date_str}  ",
            f"**Generated At:** {now_str}  ",
            f"**Total Events Logged:** {len(records)}  ",
            f"**Unique Drone Targets Tracked:** {len(tracks_summary)}  ",
            f"",
            f"---",
            f"",
            f"## 📊 Tracked Target Details",
            f""
        ]

        for tid, tinfo in sorted(tracks_summary.items()):
            first_c = tinfo["trajectory_px"][0] if tinfo["trajectory_px"] else "N/A"
            last_c = tinfo["trajectory_px"][-1] if tinfo["trajectory_px"] else "N/A"
            snapshots_str = ", ".join([f"`{s}`" for s in tinfo["snapshots"]]) if tinfo["snapshots"] else "None"

            # GPS coordinates (if available)
            geo_list = tinfo["trajectory_geo"]
            first_geo = geo_list[0] if geo_list else None
            last_geo = geo_list[-1] if geo_list else None
            first_geo_str = (
                f"`{first_geo['lat']:.5f}, {first_geo['lon']:.5f}` (dist {first_geo.get('dist_m', 0.0):.1f} m)"
                if first_geo else "N/A"
            )
            last_geo_str = (
                f"`{last_geo['lat']:.5f}, {last_geo['lon']:.5f}` (dist {last_geo.get('dist_m', 0.0):.1f} m)"
                if last_geo else "N/A"
            )

            lines.extend([
                f"### 🎯 Track ID: #{tid}",
                f"- **Detections Count:** {tinfo['count']}",
                f"- **Max Confidence:** {tinfo['max_conf']*100:.1f}%",
                f"- **First Seen:** `{tinfo['first_seen']}`",
                f"- **Last Seen:** `{tinfo['last_seen']}`",
                f"- **Initial Pixel Location:** `{first_c}`",
                f"- **Final Pixel Location:** `{last_c}`",
                f"- **Initial GPS Location:** {first_geo_str}",
                f"- **Final GPS Location:** {last_geo_str}",
                f"- **Snapshots Captured:** {snapshots_str}",
                f""
            ])

            # Optional geo trajectory table when multiple GPS records exist
            if len(geo_list) >= 2:
                lines.extend([
                    f"<details>",
                    f"<summary>GPS Trajectory ({len(geo_list)} points)</summary>",
                    f"",
                    f"| # | Timestamp | Latitude | Longitude | Alt (m) | Dist (m) | Bearing (°) |",
                    f"|---|-----------|----------|-----------|---------|----------|-------------|",
                ])
                for i, g in enumerate(geo_list, start=1):
                    ts_short = g.get("timestamp", tinfo.get("last_seen", ""))
                    lines.append(
                        f"| {i} | `{ts_short}` | {g['lat']:.6f} | {g['lon']:.6f} "
                        f"| {g.get('alt_amsl', 0.0):.1f} | {g.get('dist_m', 0.0):.1f} | "
                        f"{g.get('bearing_deg', 0.0):.1f} |"
                    )
                lines.extend([
                    f"</details>",
                    f""
                ])

        lines.extend([
            f"---",
            f"*Report generated automatically by Drone Detect Pipeline System.*"
        ])

        try:
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write("\n".join(lines))
            logger.info(f"Summary report successfully generated at: {report_path.absolute()}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to write summary report to '{report_path}': {e}")
            return None
