# Project 2 - Damage Classifier Inference Server

## Using Docker Run
```bash
docker run -it --rm -p 5000:5000 leechee1/damage-classifier
```

## Using Docker Compose

Start:
```bash
docker-compose up
```

Stop:
```bash
docker-compose down
```

## Example Requests

GET /summary:
```bash
curl localhost:5000/summary
```

POST /inference:
```bash
curl -X POST -F "image=@path/to/image.jpg" localhost:5000/inference
# example: curl -X POST -F "image=@damage/-93.795_30.03779.jpeg" localhost:5000/inference
```

Expected response:
```json
{"prediction": "damage"}
```
or
```json
{"prediction": "no_damage"}
```
