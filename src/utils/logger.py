import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import yaml

# Module logger
logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Manages structured daily event logging for drone tracking activities.
    Supports writing entries in JSON Lines (JSONL) format or raw text.
    """

    def __init__(self, config_path: str | Path):
        """
        Initializes the AuditLogger using settings from config.yaml.

        Args:
            config_path: Path to the configuration YAML file.
        """
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
        self.enabled = logging_config.get("enabled", True)
        self.log_dir = Path(logging_config.get("log_dir", "logs"))
        self.log_format = logging_config.get("log_format", "jsonl").lower()

        # Create base log directory if it does not exist
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        track_id: int,
        confidence: float,
        bbox: list[int],
        zone_status: str,
        snapshot_path: str | Path | None = None
    ):
        """
        Logs a tracking event with standardized structure.

        Args:
            track_id: Identifier of the tracked object.
            confidence: Confidence score of the detection.
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            zone_status: Geofencing zone status ("INSIDE: Zone_Name" or "OUTSIDE").
            snapshot_path: Path to the captured snapshot file if saved, else None.
        """
        if not self.enabled:
            return

        # 1. Build ISO-8601 UTC timestamp with millisecond precision
        now_utc = datetime.now(timezone.utc)
        # Format: YYYY-MM-DDTHH:MM:SS.mmmZ
        timestamp = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # 2. Build structured record dictionary
        record = {
            "timestamp": timestamp,
            "track_id": int(track_id),
            "confidence": round(float(confidence), 4),
            "bbox": [int(coord) for coord in bbox],
            "zone_status": str(zone_status),
            "snapshot_path": Path(snapshot_path).as_posix() if snapshot_path else None
        }

        # 3. Resolve output log file based on current date (daily logging)
        date_str = now_utc.strftime("%Y-%m-%d")
        ext = "jsonl" if self.log_format == "jsonl" else "log"
        log_file = self.log_dir / f"drone_events_{date_str}.{ext}"

        # 4. Write record to file
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                if self.log_format == "jsonl":
                    f.write(json.dumps(record) + "\n")
                else:
                    # Clearer text format fallback
                    snapshot_info = f", snapshot={record['snapshot_path']}" if record['snapshot_path'] else ""
                    text_line = (
                        f"[{timestamp}] ID={track_id} CONF={record['confidence']:.4f} "
                        f"BBOX={bbox} ZONE={zone_status}{snapshot_info}\n"
                    )
                    f.write(text_line)
        except Exception as e:
            logger.error(f"Failed to write event log to {log_file}: {e}")
