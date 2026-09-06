"""
MCP Server for Temperature Scaling Confidence Calibrator Skill.
"""

import json
import sys
from client import TemperatureCalibrator

CALIBRATOR = TemperatureCalibrator()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "fit_temperature",
                    "description": "Fit optimal temperature on validation logits and labels",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "validation_logits": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                            "labels": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["validation_logits", "labels"]
                    }
                },
                {
                    "name": "calibrate_logits",
                    "description": "Calibrate logits into softmax probabilities using calibrated temperature",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "logits": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["logits"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "fit_temperature":
            opt_t = CALIBRATOR.fit(args["validation_logits"], args["labels"])
            ece_res = CALIBRATOR.evaluate_ece(args["validation_logits"], args["labels"])
            return {"content": [{"type": "text", "text": json.dumps({"temperature": opt_t, "metrics": ece_res})}]}

        elif tool_name == "calibrate_logits":
            probs = CALIBRATOR.calibrate_logits(args["logits"])
            return {"content": [{"type": "text", "text": json.dumps({"calibrated_probabilities": probs})}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
