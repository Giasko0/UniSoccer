import sys
import os
import json
import subprocess
import math

def main():
    if len(sys.argv) < 2:
        print("Usage: python slice_video.py <input_video_path>")
        print("Example: python slice_video.py my_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Get base name without extension for folder and JSON
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = base_name

    # Create output folder (same name as the video, without extension)
    os.makedirs(output_folder, exist_ok=True)

    # Get video duration using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:
        duration_str = subprocess.check_output(probe_cmd).decode("utf-8").strip()
        duration = float(duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error: Could not determine video duration. Is ffmpeg/ffprobe installed?")
        print(e)
        sys.exit(1)

    if duration <= 0:
        print("Error: Video duration is zero or invalid.")
        sys.exit(1)

    # Slice into 30-second clips
    clip_duration = 30.0
    num_clips = math.ceil(duration / clip_duration)

    clips_info = []

    print(f"Processing {video_path} ({duration:.2f}s) → {num_clips} clips")

    for i in range(num_clips):
        start_time = i * clip_duration
        remaining = duration - start_time
        clip_length = min(clip_duration, remaining)

        # Name clips ascending with leading zeros: clip_001.mp4, clip_002.mp4, ...
        clip_name = f"clip_{i + 1:03d}.mp4"
        clip_full_path = os.path.join(output_folder, clip_name)

        # Fast split using stream copy (no re-encoding)
        ffmpeg_cmd = [
            "ffmpeg", "-y",                  # overwrite without asking
            "-ss", str(start_time),          # fast seek
            "-i", video_path,
            "-t", str(clip_length),
            "-c", "copy",                    # copy streams (fast)
            "-avoid_negative_ts", "make_zero",
            clip_full_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  ✓ Created {clip_name} ({clip_length:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to create {clip_name}")
            print(e.stderr.decode())
            sys.exit(1)

        # Relative path as shown in the requested JSON format
        relative_path = f"{output_folder}/{clip_name}"

        clips_info.append({
            "video": relative_path,
            "caption": "",                    # empty as per structure (example showed "clearance" as placeholder)
            "comments_text": "",
            "comments_text_anonymized": ""
        })

    # Save JSON file next to the folder (named after the original video)
    json_filename = f"{base_name}_clips.json"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(clips_info, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Done!")
    print(f"   Folder: ./{output_folder}/")
    print(f"   Clips:  {num_clips} × 30s clips")
    print(f"   JSON:   ./{json_filename}  ← contains {len(clips_info)} entries")


if __name__ == "__main__":
    main()