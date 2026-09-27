# Stage A — Example tiles and VLM responses

Ablation: `no_lsi_numbers` (repeat 1)  
Backend: `ollama` / `qwen2.5vl`

## Example 1 — point 6

- **target**: `1`
- **VLM class / score**: `high` / `85.0`
- **imagery**: `2020-11-01` → `2024-02-28` (event_date=None, post_event=None)
- **composite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_006_x474143_y2624656_composite.png`
- **satellite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_006_x474143_y2624656_satellite.png`
- **include_lsi**: `False` (sat_ok=False)

```json
{
  "class": "high",
  "score": 85,
  "key_factors": [
    "steep slope",
    "high rainfall",
    "sonker TRI 5"
  ],
  "reason": "The steep slope and high rainfall indicate a high risk of landslides, exacerbated by the rough terrain and lack of vegetation cover."
}
```

## Example 2 — point 0

- **target**: `0`
- **VLM class / score**: `moderate` / `50.0`
- **imagery**: `2020-11-01` → `2024-02-28` (event_date=None, post_event=None)
- **composite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_000_x498435_y2626185_composite.png`
- **satellite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_000_x498435_y2626185_satellite.png`
- **include_lsi**: `False` (sat_ok=False)

```json
{
  "class": "moderate",
  "score": 50,
  "key_factors": [
    "steep slope",
    "high rainfall",
    "sonker terrain"
  ],
  "reason": "The steep slope and high rainfall indicate potential for shallow landslides, while the sonker terrain suggests roughness and poor drainage."
}
```

## Example 3 — point 7

- **target**: `1`
- **VLM class / score**: `moderate` / `65.0`
- **imagery**: `2020-11-01` → `2024-02-28` (event_date=2009-05-10, post_event=True)
- **composite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_007_x471590_y2624892_composite.png`
- **satellite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_007_x471590_y2624892_satellite.png`
- **include_lsi**: `False` (sat_ok=False)

```json
{
  "class": "moderate",
  "score": 65,
  "key_factors": [
    "steep slope",
    "high rainfall",
    "sonker TRI 5"
  ],
  "reason": "The steep slope and high rainfall indicate potential for shallow debris slides, while the rugged terrain and sonker TRI 5 suggest increased susceptibility."
}
```

## Example 4 — point 1

- **target**: `0`
- **VLM class / score**: `moderate` / `50.0`
- **imagery**: `2020-11-01` → `2024-02-28` (event_date=None, post_event=None)
- **composite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_001_x483135_y2634885_composite.png`
- **satellite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_001_x483135_y2634885_satellite.png`
- **include_lsi**: `False` (sat_ok=False)

```json
{
  "class": "moderate",
  "score": 50,
  "key_factors": [
    "steep slope",
    "high rainfall",
    "soil instability"
  ],
  "reason": "The steep slope and high rainfall indicate potential for shallow landslides, while soil instability further increases susceptibility."
}
```

## Example 5 — point 8

- **target**: `1`
- **VLM class / score**: `high` / `85.0`
- **imagery**: `2020-11-01` → `2024-02-28` (event_date=None, post_event=None)
- **composite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_008_x501733_y2619530_composite.png`
- **satellite**: `C:/Users/sharm/lsi-ml-pipeline-1/outputs_vlm/tiles/no_lsi/point_008_x501733_y2619530_satellite.png`
- **include_lsi**: `False` (sat_ok=False)

```json
{
  "class": "high",
  "score": 85,
  "key_factors": [
    "steep slope",
    "high rainfall",
    "sonker terrain"
  ],
  "reason": "The steep slope and high rainfall indicate a high risk of shallow debris slides, while the sonker terrain adds to the instability."
}
```
