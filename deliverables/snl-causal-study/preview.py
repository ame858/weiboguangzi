"""Produce a side-by-side preview from existing development predictions."""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiment import filter_by_config, load_candidates, process, summarize, torch, with_nms
from snl_inputs import DEFAULT_OUTPUT, DEVELOPMENT, HEIGHT, WIDTH, load_gray, load_labels, read_json, write_csv, write_json
from videosar.snl_causal import detection_metrics, matched_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"gpu")
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--fps",type=int,choices=range(1, 31),default=15)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    choice = read_json(output/"development-status.json")["best"]
    config = choice["config"]
    gray = load_gray(output,args.ffmpeg)
    labels,_,_ = load_labels(output,DEVELOPMENT,gray)
    if config["method"] in ("gray", "temporal"):
        cached = with_nms(load_candidates(output, config, "development"), config["nms"])
        predicted = filter_by_config(cached, config)
    else:
        predicted = process(output, config, gray, labels, "development")
    verified, _ = summarize(predicted, labels)
    if verified != choice["development"]:
        raise ValueError("Saved development predictions do not match the selected decision")
    width, height = WIDTH*2, HEIGHT+30
    temporary = output/"development-preview.partial.mp4"
    command = [args.ffmpeg,"-y","-v","error","-f","rawvideo","-pixel_format","rgb24",
               "-video_size",f"{width}x{height}","-framerate",str(args.fps),"-i","-","-an",
               "-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(temporary)]
    summary = np.zeros(3, dtype=int)
    rows = []
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for i, targets in labels.items():
            boxes,scores,_ = predicted[i]
            matching = matched_boxes(boxes,targets)
            accepted = set(matching[:,0])
            covered = set(matching[:,1])
            summary += (len(matching), len(boxes) - len(matching), len(targets) - len(matching))
            image = Image.fromarray(gray[i-2]).convert("RGB")
            panel = Image.new("RGB",(width,height),"#131b23")
            panel.paste(image,(0,30))
            panel.paste(image,(WIDTH,30))
            draw = ImageDraw.Draw(panel)
            draw.text((10,8),f"Development frame {i+149} | Annotation | Amber: missed",fill="white")
            draw.text((WIDTH+10,8),f"Prediction | TP {len(matching)} / FP {len(boxes)-len(matching)} / FN {len(targets)-len(matching)}",fill="white")
            for j, box in enumerate(targets):
                draw.rectangle(tuple(box+[0,30,0,30]),outline="#4eff9f" if j in covered else "#ffbd51",width=2)
            for j, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                draw.rectangle(tuple(box+[WIDTH,30,WIDTH,30]),
                               outline="#4eff9f" if j in accepted else "#ff6a59",width=2)
                rows.append(dict(annotation_id=i, video_frame=i+148, x0=float(box[0]), y0=float(box[1]),
                                 x1=float(box[2]), y1=float(box[3]), score=float(score), matched=j in accepted))
            encoder.stdin.write(panel.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg failed to encode the development preview")
    destination=output/"development-preview.mp4"
    temporary.replace(destination)
    predictions = output/"development-preview-predictions.csv"
    write_csv(predictions, rows, ("annotation_id", "video_frame", "x0", "y0", "x1", "y1", "score", "matched"))
    write_json(output/"preview-receipt.json", {"config": config, "development": detection_metrics(summary, len(labels)),
                                             "video": destination.name, "fps": args.fps, "predictions": predictions.name})
    print(f"Encoded {len(labels)} development frames: {destination}",flush=True)


if __name__ == "__main__":
    main()
